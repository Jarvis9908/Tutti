/**
 * coordinator.cpp -- Coordinator orchestration impl (R9.2 / R9.3).
 *
 * Brings up registry -> nvme_storage -> block_storage -> memory ->
 * io_engine in dependency order and exposes thin passthroughs to the
 * high-level data-plane operations.
 */

// NOTE on include order: the subsystem headers below transitively
// pull libnvm's <nvm_types.h>, which must be parsed while the global
// namespace still has the plain C <stdint.h> typedefs (uint8_t etc).
// coordinator.h includes <cuda_runtime.h>, which on CUDA 13 drags in
// libcu++ shims that shadow those typedefs -- so we include the
// nvm_types-pulling headers FIRST (mirroring memory/src ordering)
// and coordinator.h LAST.
#include "../../device_manager/include/local_nvme_direct_registry.h"
#include "../../device_manager/include/nvmeservice_backed_registry.h"
#include "../../nvme_storage/include/host_fs_backed_nvme_storage.h"
#include "../../nvme_storage/include/nvme_file_device_handle.h"   // sizeof() for AUTO budget-based handle_l1/l2_capacity
#include "../../block_storage/include/host_fs_backed_block_storage.h"
#include "../../block_storage/include/block_storage.h"
#include "../../memory/include/host_device_memory_subsystem.h"
#include "../../memory/include/memory_kind.h"
#include "../../io_engine/include/local_nvme/local_nvme_io_engine.h"
#include "../../coordinator/include/device.h"

#include "coordinator.h"

#include <algorithm>
#include <cstdio>
#include <utility>

namespace tutti {

Coordinator::Coordinator()  = default;
Coordinator::~Coordinator() { shutdown(); }

IIoEngine* Coordinator::io_engine() const {
    return engine_.get();
}

INvmeStorage::CacheStats Coordinator::cache_stats() const {
    if (!storage_) return {};
    return storage_->cache_stats();
}

bool Coordinator::bootstrap(const CoordinatorConfig& cfg) {
    if (booted_) {
        std::fprintf(stderr,
            "[coordinator] bootstrap: already booted; call shutdown() first\n");
        return false;
    }

    // AUTO capacity derivation: handle_l1_capacity/handle_l2_capacity
    // == 0 means "derive from a GPU/host memory budget instead of a
    // magic number" -- see coordinator_config.h's doc comment for the
    // full derivation.  kGpuFileMaxShards is used as the conservative
    // per-file shard multiplier since the actual shard count varies
    // per GpuFile and isn't known at bootstrap time.
    const std::size_t per_file_bytes =
        (sizeof(NvmeFileDeviceHandle) + sizeof(void*)) * (std::size_t)kGpuFileMaxShards;

    if (cfg.handle_l1_capacity != 0) {
        handle_l1_capacity_ = cfg.handle_l1_capacity;
    } else {
        const uint64_t budget = cfg.handle_l1_gpu_budget_bytes != 0
                               ? cfg.handle_l1_gpu_budget_bytes
                               : (512ull << 20);
        std::size_t computed = (std::size_t)(budget / per_file_bytes);
        if (computed == 0) computed = 1;
        computed = std::min<std::size_t>(computed, 0xFFFFFFFFull);
        handle_l1_capacity_ = (uint32_t)computed;
        std::fprintf(stderr,
            "[coordinator] handle_l1_capacity AUTO = %u files (budget=%llu "
            "B / per_file=%zu B [kGpuFileMaxShards=%u * "
            "(sizeof(NvmeFileDeviceHandle)=%zu B + sizeof(void*)=%zu B)])\n",
            handle_l1_capacity_, (unsigned long long)budget, per_file_bytes,
            (unsigned)kGpuFileMaxShards, sizeof(NvmeFileDeviceHandle),
            sizeof(void*));
    }

    if (cfg.handle_l2_capacity != 0) {
        handle_l2_capacity_ = cfg.handle_l2_capacity;
    } else {
        const uint64_t budget = cfg.handle_l2_host_budget_bytes != 0
                               ? cfg.handle_l2_host_budget_bytes
                               : (2ull << 30);
        std::size_t computed = (std::size_t)(budget / per_file_bytes);
        if (computed == 0) computed = 1;
        computed = std::min<std::size_t>(computed, 0xFFFFFFFFull);
        handle_l2_capacity_ = (uint32_t)computed;
        std::fprintf(stderr,
            "[coordinator] handle_l2_capacity AUTO = %u files (budget=%llu "
            "B / per_file=%zu B)\n",
            handle_l2_capacity_, (unsigned long long)budget, per_file_bytes);
    }
    if (handle_l2_capacity_ < handle_l1_capacity_) {
        std::fprintf(stderr,
            "[coordinator] bootstrap: handle_l2_capacity=%u < "
            "handle_l1_capacity=%u -- L1 must fit inside L2's backing "
            "store; raising L2 to match\n",
            handle_l2_capacity_, handle_l1_capacity_);
        handle_l2_capacity_ = handle_l1_capacity_;
    }

    // ---- 1. Device registry (mode-specific) ------------------------
    //   Both branches produce an IDeviceRegistry whose Open() has
    //   succeeded; everything below the registry consumes only
    //   `const Device*` and is mode-agnostic.
    if (cfg.mode == CoordinatorMode::IN_PROCESS) {
        if (cfg.pci_addrs.empty()) {
            std::fprintf(stderr,
                "[coordinator] bootstrap: IN_PROCESS mode needs "
                "cfg.pci_addrs\n");
            return false;
        }
        std::vector<LocalNvmeDirectConfig> dcfgs;
        dcfgs.reserve(cfg.pci_addrs.size());
        for (const auto& bdf : cfg.pci_addrs) {
            LocalNvmeDirectConfig c{};
            c.pci_addr          = bdf;
            c.kernel_ioq_cap    = cfg.kernel_ioq_cap;
            c.build_queue_group = true;                 // GPU submit needs d_qps
            c.cuda_device       = cfg.cuda_device;
            c.num_user_queues   = cfg.num_user_queues_per_device;
            c.queue_depth       = cfg.queue_depth;
            c.namespace_id      = cfg.namespace_id;
            dcfgs.push_back(std::move(c));
        }
        registry_ =
            std::make_unique<LocalNvmeDirectRegistry>(std::move(dcfgs));
    } else {  // CoordinatorMode::SERVICE_CLIENT
        if (cfg.daemon_endpoint.empty() || cfg.daemon_device_ids.empty()) {
            std::fprintf(stderr,
                "[coordinator] bootstrap: SERVICE_CLIENT mode needs "
                "cfg.daemon_endpoint + cfg.daemon_device_ids\n");
            return false;
        }
        std::vector<NvmeServiceBackedRequest> reqs;
        reqs.reserve(cfg.daemon_device_ids.size());
        for (int32_t id : cfg.daemon_device_ids) {
            NvmeServiceBackedRequest r{};
            r.daemon_device_id  = id;
            r.cuda_device       = cfg.cuda_device;
            // Lease quota request == the queue-group width we intend
            // to build on our own attach_client fd.
            r.num_queues        = (int32_t)cfg.num_user_queues_per_device;
            r.build_queue_group = true;                 // GPU submit needs d_qps
            r.num_user_queues   = cfg.num_user_queues_per_device;
            r.queue_depth       = cfg.queue_depth;
            r.namespace_id      = cfg.namespace_id;
            reqs.push_back(std::move(r));
        }
        registry_ = std::make_unique<NvmeServiceBackedRegistry>(
            cfg.daemon_endpoint, std::move(reqs));
    }

    if (!registry_->Open()) {
        std::fprintf(stderr, "[coordinator] bootstrap: registry Open() failed\n");
        shutdown();
        return false;
    }

    devices_.clear();
    const std::size_t n = registry_->device_count();
    devices_.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const Device* d = registry_->device_at(i);
        if (d == nullptr) {
            std::fprintf(stderr,
                "[coordinator] bootstrap: device_at(%zu) is null\n", i);
            shutdown();
            return false;
        }
        devices_.push_back(d);
    }

    // ---- 2. nvme_storage -------------------------------------------
    storage_ = std::make_unique<HostFsBackedNvmeStorage>();
    if (!storage_->bootstrap(devices_)) {
        std::fprintf(stderr, "[coordinator] bootstrap: nvme_storage failed\n");
        shutdown();
        return false;
    }

    // ---- 3. block_storage ------------------------------------------
    block_ = std::make_unique<HostFsBackedBlockStorage>();
    if (!block_->bootstrap(storage_.get(), devices_)) {
        std::fprintf(stderr, "[coordinator] bootstrap: block_storage failed\n");
        shutdown();
        return false;
    }

    // R11.3: size the two-tier handle caches (block_storage's own
    // ShardPtrSlot pool + nvme_storage's per-shard two-tier
    // NvmeFileDeviceHandle cache, forwarded internally -- see
    // block_storage.h's configure_handle_pool doc) BEFORE any
    // acquire_device_handle / handle_for call.
    if (!block_->configure_handle_pool(handle_l1_capacity_, handle_l2_capacity_)) {
        std::fprintf(stderr,
            "[coordinator] bootstrap: configure_handle_pool(l1=%u, l2=%u) "
            "failed\n", handle_l1_capacity_, handle_l2_capacity_);
        shutdown();
        return false;
    }

    // ---- 4. memory (R7) --------------------------------------------
    mem_ = std::make_unique<HostDeviceMemorySubsystem>();
    mem_->set_descriptor_format(cfg.descriptor_format);
    mem_->configure_prp_pool(cfg.prp_l1_gpu_budget_bytes,
                              cfg.prp_l2_host_budget_bytes);
    mem_->bind_devices(devices_);

    // ---- 5. io_engine (R8.3) ---------------------------------------
    engine_ = std::make_unique<LocalNvmeIoEngine>(
        mem_.get(), cfg.max_entries_per_batch);

    booted_ = true;
    return true;
}

void Coordinator::shutdown() {
    // Reverse dependency order.  Each step is null-safe so a partial
    // bootstrap rollback lands here too.  Stream 0: teardown has no
    // "caller's stream" to order against; correctness here relies on
    // the fact that nothing should still be submitting IO by the time
    // shutdown() runs (same assumption the rest of this method already
    // makes when it resets mem_/engine_ unconditionally).
    drop_cached_handles(/*stream=*/0);   // release acquired handles while block_ alive
    gpu_files_by_id_.clear();
    engine_.reset();              // d_scratch cudaFree
    mem_.reset();                 // drops DMA maps + IO-slice tables
    if (block_)   { block_->shutdown();   block_.reset(); }
    if (storage_) { storage_->shutdown(); storage_.reset(); }
    if (registry_) { registry_->Close();  registry_.reset(); }
    devices_.clear();
    booted_ = false;
}

// ---------------------------------------------------------------------------
// GpuFile lifecycle
// ---------------------------------------------------------------------------

GpuFile* Coordinator::open_gpu_file(const GpuFileSpec& spec, uint32_t flags) {
    if (!block_) return nullptr;
    GpuFile* gf = block_->open_gpu_file(spec, flags);
    if (gf != nullptr) gpu_files_by_id_[gf->id] = gf;
    return gf;
}

std::vector<GpuFile*> Coordinator::open_gpu_files_batch(
    const GpuFileSpec* specs, uint32_t count, uint32_t flags)
{
    if (!block_ || count == 0) return {};
    auto results = block_->open_gpu_files_batch(specs, count, flags);
    for (auto* gf : results)
        if (gf != nullptr) gpu_files_by_id_[gf->id] = gf;
    return results;
}

bool Coordinator::has_gpu_file(GpuFileId id) const {
    return gpu_files_by_id_.find(id) != gpu_files_by_id_.end();
}

bool Coordinator::delete_gpu_file(GpuFile* gf, bool persist_now, cudaStream_t stream) {
    if (!block_) return false;
    if (gf != nullptr) {
        // Genuinely release (not just free host memory) any handle
        // still cached for this id -- the file is going away, so the
        // GPU-side residency hint is correct here (unlike the
        // refresh-in-place case handle_for/handle_for_batch handle
        // themselves).
        auto it = last_handle_.find(gf->id);
        if (it != last_handle_.end()) {
            block_->release_device_handle(it->second, reinterpret_cast<GpuStreamHandle>(stream));
            last_handle_.erase(it);
        }
        gpu_files_by_id_.erase(gf->id);
    }
    return block_->delete_gpu_file(gf, persist_now);
}

bool Coordinator::delete_gpu_files_batch(GpuFile* const* files, uint32_t count,
                                         bool* out_ok, cudaStream_t stream) {
    if (!block_) return false;
    if (count == 0) return true;
    if (files == nullptr || out_ok == nullptr) return false;

    // Capture every id BEFORE calling block_->delete_gpu_files_batch:
    // that call erases the GpuFile objects from block_storage's
    // internal map (unique_ptr destruction), so `files[i]` is a
    // dangling pointer for every successfully-deleted entry once it
    // returns -- reading files[i]->id afterward would be a
    // use-after-free (mirrors delete_gpu_file's single-item ordering,
    // which reads gf->id BEFORE calling block_->delete_gpu_file).
    std::vector<GpuFileId> ids(count, 0);
    for (uint32_t i = 0; i < count; ++i)
        if (files[i] != nullptr) ids[i] = files[i]->id;

    if (!block_->delete_gpu_files_batch(files, count, out_ok)) {
        // Fall through -- still clean up bookkeeping for whichever
        // ids DID succeed (out_ok is filled per-item regardless).
    }
    for (uint32_t i = 0; i < count; ++i) {
        if (files[i] == nullptr || !out_ok[i]) continue;
        auto it = last_handle_.find(ids[i]);
        if (it != last_handle_.end()) {
            block_->release_device_handle(it->second, reinterpret_cast<GpuStreamHandle>(stream));
            last_handle_.erase(it);
        }
        gpu_files_by_id_.erase(ids[i]);
    }
    bool all_ok = true;
    for (uint32_t i = 0; i < count; ++i) if (!out_ok[i]) all_ok = false;
    return all_ok;
}

GpuFileHandle* Coordinator::acquire_device_handle(GpuFile* gf, cudaStream_t stream) {
    if (!block_) return nullptr;
    return block_->acquire_device_handle(gf, reinterpret_cast<GpuStreamHandle>(stream));
}

void Coordinator::release_device_handle(GpuFileHandle* h, cudaStream_t stream) {
    if (block_) block_->release_device_handle(h, reinterpret_cast<GpuStreamHandle>(stream));
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

MemoryRegion* Coordinator::allocate_device(std::size_t size,
                                           MemoryKind  kind,
                                           int         device_id) {
    if (!mem_) return nullptr;
    return mem_->allocate_device(size, kind, device_id);
}

MemoryRegion* Coordinator::register_tensor(const TensorRegistrationSpec& spec) {
    if (!mem_) return nullptr;
    MemoryRegion* r = mem_->register_tensor(spec);
    if (r == nullptr) {
        const char* err = mem_->last_register_error();
        std::fprintf(stderr,
            "[coordinator] register_tensor failed: %s\n",
            err ? err : "(unknown)");
    }
    return r;
}

void Coordinator::free(MemoryRegion* region) {
    if (mem_) mem_->free(region);
}

uint32_t Coordinator::batch_entry_count(MemoryRegion* region) const {
    if (!mem_ || region == nullptr) return 0;
    uint32_t n = 0;
    for (const IoSliceView& v : mem_->list_io_slices(region)) n += v.num_ios;
    return n;
}

// ---------------------------------------------------------------------------
// Transparent handle cache (id-keyed; R11.3)
// ---------------------------------------------------------------------------

GpuFileHandle* Coordinator::handle_for(GpuFileId id, cudaStream_t stream) {
    if (!block_) return nullptr;

    auto fit = gpu_files_by_id_.find(id);
    if (fit == gpu_files_by_id_.end()) {
        std::fprintf(stderr,
            "[coordinator] handle_for(id=%u): not created/opened this "
            "session (call open_gpu_file first)\n", id);
        return nullptr;
    }

    // Always ensure-resident (near-free on a hit inside
    // block_storage/nvme_storage's own two-tier caches) rather than
    // trusting a Coordinator-side cache -- the lower layers' own LRUs
    // may have evicted/recycled the GPU slot a previous call's handle
    // pointed at.
    GpuFileHandle* h = block_->acquire_device_handle(
        fit->second, reinterpret_cast<GpuStreamHandle>(stream));
    if (h == nullptr) {
        std::fprintf(stderr,
            "[coordinator] handle_for(id=%u): acquire_device_handle failed\n", id);
        return nullptr;
    }

    // Free the PREVIOUS host-side handle object for this id now that
    // a fresh one is in hand -- plain host memory reclaim only (see
    // coordinator.h's doc comment: NOT routed through
    // block_->release_device_handle, which would incorrectly hint
    // eviction of the slot `h` just re-filled for this same id).
    auto old = last_handle_.find(id);
    if (old != last_handle_.end()) {
        delete old->second;
        old->second = h;
    } else {
        last_handle_[id] = h;
    }
    return h;
}

bool Coordinator::handle_for_batch(const GpuFileId* ids, uint32_t count,
                                   cudaStream_t stream, GpuFileHandle** out) {
    if (!block_) return false;
    if (count == 0) return true;
    if (ids == nullptr || out == nullptr) return false;
    if (count > handle_l1_capacity_) {
        std::fprintf(stderr,
            "[coordinator] handle_for_batch: count=%u > "
            "handle_l1_capacity=%u -- a single batch's distinct-file "
            "working set must fit under the cap (raise "
            "cfg.handle_l1_capacity)\n", count, handle_l1_capacity_);
        return false;
    }

    std::vector<GpuFile*> files(count, nullptr);
    for (uint32_t i = 0; i < count; ++i) {
        auto fit = gpu_files_by_id_.find(ids[i]);
        if (fit == gpu_files_by_id_.end()) {
            std::fprintf(stderr,
                "[coordinator] handle_for_batch(id=%u): not created/opened "
                "this session (call open_gpu_file first)\n",
                ids[i]);
            return false;
        }
        files[i] = fit->second;
    }

    if (!block_->acquire_device_handles_batch(
            files.data(), count, reinterpret_cast<GpuStreamHandle>(stream), out)) {
        std::fprintf(stderr,
            "[coordinator] handle_for_batch: acquire_device_handles_batch "
            "failed (count=%u)\n", count);
        return false;
    }

    for (uint32_t i = 0; i < count; ++i) {
        auto old = last_handle_.find(ids[i]);
        if (old != last_handle_.end()) {
            delete old->second;
            old->second = out[i];
        } else {
            last_handle_[ids[i]] = out[i];
        }
    }
    return true;
}

bool Coordinator::sync_file(GpuFileId id, cudaStream_t stream) {
    // R12 B-1 / 隐患-2: synchronize the GPU stream first so that
    // GPU kernels submitting NVMe write commands on `stream` have
    // actually completed (commands submitted + CQ polled).  Without
    // this, fsync could run before the writes are even submitted.
    if (stream != nullptr) {
        cudaError_t err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            std::fprintf(stderr,
                "[coordinator] sync_file(id=%u): cudaStreamSynchronize "
                "failed: %s\n", id, cudaGetErrorString(err));
            return false;
        }
    }

    if (!storage_) return false;
    auto fit = gpu_files_by_id_.find(id);
    if (fit == gpu_files_by_id_.end()) {
        std::fprintf(stderr,
            "[coordinator] sync_file(id=%u): unknown id\n", id);
        return false;
    }
    // fsync triggers an NVMe FLUSH via the block layer (flushes the
    // controller's volatile write cache, including GPU-written data).
    // Data itself bypasses page cache (GPU-direct DMA / O_DIRECT), so
    // there are no dirty data pages to flush -- the metadata is
    // normally already durable via the create/flush_metadata path.
    bool ok = true;
    for (NvmeFile* s : fit->second->shards)
        if (!storage_->sync(s)) ok = false;
    return ok;
}

void Coordinator::drop_cached_handles(cudaStream_t stream) {
    if (block_) {
        for (auto& kv : last_handle_)
            block_->release_device_handle(kv.second, reinterpret_cast<GpuStreamHandle>(stream));
    }
    last_handle_.clear();
}

// ---------------------------------------------------------------------------
// IO submission
// ---------------------------------------------------------------------------

bool Coordinator::submit_batch(const std::vector<NvmeBatchInputTensor>& inputs,
                               bool                                     is_read,
                               cudaStream_t                             stream) {
    if (!engine_) {
        std::fprintf(stderr,
            "[coordinator] submit_batch: not booted (no io_engine)\n");
        return false;
    }
    return engine_->submit_batch(inputs, is_read, stream);
}

bool Coordinator::submit_batch(MemoryRegion*  tensor_region,
                               GpuFileHandle* file_handle,
                               uint64_t       file_byte_offset,
                               bool           is_read,
                               cudaStream_t   stream) {
    std::vector<NvmeBatchInputTensor> inputs;
    inputs.push_back({tensor_region, file_handle, file_byte_offset});
    return submit_batch(inputs, is_read, stream);
}

} // namespace tutti
