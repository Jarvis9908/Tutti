/**
 * host_fs_backed_nvme_storage_device.cu -- GPU-side bring-up + tear-
 *                                          down of NvmeFileDeviceHandle.
 *
 * Layer: nvme_storage (R5b; two-tier async pooled R11.3).
 *
 * Split out from host_fs_backed_nvme_storage.cpp so the host-side
 * file (which is .cpp) doesn't have to start dragging in CUDA
 * runtime.  Everything else still lives in the .cpp file; this
 * unit only knows about cudaMalloc / cudaMemcpy / cudaFree /
 * cudaLaunchHostFunc (transitively, via TieredHandleCache).
 *
 * R11.3: handles now live in a per-cuda-device
 * TieredHandleCache<NvmeFileDeviceHandle, NvmeFileId> (CPU-pinned L2
 * backing a small GPU-resident L1) instead of a single-tier
 * GpuSlotPool -- see nvme_storage.h's updated doc comment and
 * doc/tutti_vs_geminifs_rw_and_integration.md for the full
 * rationale.  acquire_device_handle is now "ensure resident"
 * (idempotent); release_device_handle takes the NvmeFile* itself
 * (not the recycled GPU pointer) and is an advisory "may downgrade
 * now" hint.
 *
 * The rare heavily-fragmented file's overflow extent buffer is still
 * a genuine per-file cudaMalloc (not tiered -- rare + variably sized,
 * not worth it), tracked by NvmeFileId (stable) instead of by handle
 * pointer (which now gets recycled across L1 slots).
 */

#include "host_fs_backed_nvme_storage.h"

#include "nvme_file.h"
#include "nvme_file_device_handle.h"
#include "nvme_file_header.h"           // kNvmeFileHeaderMaxExtents (extent cap)
#include "../../coordinator/include/device.h"
#include "../../device_manager/include/local_nvme_device.h"
#include "../../device_manager/include/nvme_queue_group.h"   // R5b

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

namespace tutti {

namespace {

// Freed once the async H2D copy that reads it has actually executed
// (queued right after the cudaMemcpyAsync that uses it).  This is NOT
// the overflow device buffer itself (that lives until the file is
// erased from the cache) -- just the throwaway pinned host staging.
struct OverflowStagingFreeCtx { void* pinned_host_ptr; };

void CUDART_CB free_overflow_staging(void* data) {
    auto* ctx = static_cast<OverflowStagingFreeCtx*>(data);
    if (ctx->pinned_host_ptr != nullptr) cudaFreeHost(ctx->pinned_host_ptr);
    delete ctx;
}

// Freed when the file is erased from the cache (delete_file), ordered
// on the same stream as that erase so it never races a kernel that
// might still be reading through extents_overflow.
struct OverflowDeviceFreeCtx { void* device_ptr; };

void CUDART_CB free_overflow_device(void* data) {
    auto* ctx = static_cast<OverflowDeviceFreeCtx*>(data);
    if (ctx->device_ptr != nullptr) cudaFree(ctx->device_ptr);
    delete ctx;
}

// Best-effort check that the CALLER has the right CUDA device
// current before we touch `expected_cuda_device`'s pool.  The CUDA
// Runtime API has no portable way to ask "which device was `stream`
// created under", so this checks the proxy this codebase already
// relies on everywhere (see coordinator.h's doc comments): the
// caller is expected to cudaSetDevice(cfg.cuda_device) -- and use a
// stream created under that same context -- before calling in here.
// Catches the common "forgot to cudaSetDevice" / "mixed up which GPU
// this file lives on" misuse with one clear diagnostic instead of a
// cryptic CUDA error surfacing later inside a cudaMemcpyAsync/
// cudaEventRecord deep in GpuSlotPool.
bool check_current_device_(const char* who, int expected_cuda_device) {
    int cur = -1;
    cudaError_t cerr = cudaGetDevice(&cur);
    if (cerr != cudaSuccess) {
        std::fprintf(stderr, "[nvme_storage] %s: cudaGetDevice failed: %s\n",
                     who, cudaGetErrorString(cerr));
        return false;
    }
    if (cur != expected_cuda_device) {
        std::fprintf(stderr,
            "[nvme_storage] %s: current CUDA device=%d != this file's "
            "queue_group cuda_device=%d -- caller must cudaSetDevice(%d) "
            "(and pass a stream created under that same device context) "
            "before this call\n",
            who, cur, expected_cuda_device, expected_cuda_device);
        return false;
    }
    return true;
}

} // namespace

bool HostFsBackedNvmeStorage::configure_handle_pool(uint32_t l1_capacity, uint32_t l2_capacity)
{
    std::lock_guard<std::mutex> lock(metadata_caches_mtx_);
    if (!metadata_caches_.empty()) {
        std::fprintf(stderr,
            "[nvme_storage] configure_handle_pool: called after at least "
            "one cache was already lazily initialised (i.e. after the "
            "first acquire_device_handle) -- must be called right after "
            "bootstrap(), before any acquire\n");
        return false;
    }
    if (l1_capacity == 0 || l2_capacity == 0) {
        std::fprintf(stderr,
            "[nvme_storage] configure_handle_pool: l1_capacity=%u "
            "l2_capacity=%u (both must be > 0)\n", l1_capacity, l2_capacity);
        return false;
    }
    l1_capacity_ = l1_capacity;
    l2_capacity_ = l2_capacity;
    return true;
}

TieredHandleCache<NvmeFileDeviceHandle, NvmeFileId>*
HostFsBackedNvmeStorage::get_or_init_metadata_cache_(int cuda_device)
{
    std::lock_guard<std::mutex> lock(metadata_caches_mtx_);
    auto it = metadata_caches_.find(cuda_device);
    if (it != metadata_caches_.end()) return it->second.get();

    if (l1_capacity_ == 0) {
        std::fprintf(stderr,
            "[nvme_storage] get_or_init_metadata_cache_: configure_handle_pool was "
            "never called\n");
        return nullptr;
    }

    auto cache = std::make_unique<TieredHandleCache<NvmeFileDeviceHandle, NvmeFileId>>();
    typename TieredHandleCache<NvmeFileDeviceHandle, NvmeFileId>::Config cfg{};
    cfg.l1_capacity = l1_capacity_;
    cfg.l2_capacity = l2_capacity_;
    cfg.cuda_device = cuda_device;
    if (!cache->init(cfg)) {
        std::fprintf(stderr,
            "[nvme_storage] get_or_init_metadata_cache_: TieredHandleCache::init("
            "l1=%u, l2=%u, cuda_device=%d) failed\n",
            l1_capacity_, l2_capacity_, cuda_device);
        return nullptr;
    }
    auto* raw = cache.get();
    metadata_caches_.emplace(cuda_device, std::move(cache));
    return raw;
}

bool HostFsBackedNvmeStorage::build_handle_template_(
    NvmeFile* file, cudaStream_t stream, NvmeFileDeviceHandle* out)
{
    if (file == nullptr || file->device == nullptr) return false;

    auto* bp = static_cast<LocalNvmeDevice*>(file->device->backend_private);
    if (bp == nullptr) {
        std::fprintf(stderr,
            "[nvme_storage] build_handle_template_: device has no "
            "LocalNvmeDevice payload\n");
        return false;
    }
    if (!bp->queue_group) {
        std::fprintf(stderr,
            "[nvme_storage] build_handle_template_: device(id=%d) has no "
            "queue_group. The registry that produced this device must "
            "have been opened with build_queue_group=true.\n",
            file->device->device_id);
        return false;
    }
    NvmeQueueGroup* qg = bp->queue_group.get();
    if (qg->d_qps() == nullptr || qg->n_qps() == 0) {
        std::fprintf(stderr,
            "[nvme_storage] build_handle_template_: queue_group has no "
            "d_qps (n_qps=%u, d_qps=%p)\n",
            (unsigned)qg->n_qps(), (void*)qg->d_qps());
        return false;
    }
    if (file->extents.empty()) {
        std::fprintf(stderr,
            "[nvme_storage] build_handle_template_: file has no extents\n");
        return false;
    }

    std::memset(out, 0, sizeof(*out));
    out->file_id              = file->id;
    out->logical_size_bytes   = file->size_bytes;
    out->header_bytes         = 0;   // R11.5: no in-band header prefix
    out->nvme_block_size      = bp->blk_size;
    out->nvme_block_size_log  = bp->blk_size_log;
    out->namespace_id         = bp->namespace_id;
    out->num_extents          = (uint32_t)file->extents.size();
    if (out->num_extents > kNvmeFileHeaderMaxExtents) {
        std::fprintf(stderr,
            "[nvme_storage] build_handle_template_: too many extents "
            "(%u > %u)\n", (unsigned)out->num_extents,
            (unsigned)kNvmeFileHeaderMaxExtents);
        return false;
    }

    // Inline extents -- the common case (see nvme_file_device_handle.h
    // sizing note): files with <= kNvmeFileDeviceHandleInlineExtents
    // never touch the overflow path below.
    const uint32_t n_inline =
        out->num_extents < kNvmeFileDeviceHandleInlineExtents
            ? out->num_extents
            : kNvmeFileDeviceHandleInlineExtents;
    for (uint32_t i = 0; i < n_inline; ++i) out->extents[i] = file->extents[i];
    const uint32_t n_overflow =
        out->num_extents > kNvmeFileDeviceHandleInlineExtents
            ? out->num_extents - kNvmeFileDeviceHandleInlineExtents
            : 0;

    out->d_qps     = qg->d_qps();
    out->num_d_qps = qg->n_qps();
    out->extents_overflow = nullptr;

    if (n_overflow == 0) return true;

    // Overflow buffer -- only for the rare heavily-fragmented file.
    // Genuinely per-file cudaMalloc (not tiered), but the fill is
    // still async: a throwaway pinned staging buffer + cudaMemcpyAsync,
    // with the staging buffer's cudaFreeHost deferred to a host
    // callback queued right after the copy.
    void* overflow_dev = nullptr;
    cudaError_t cerr = cudaMalloc(&overflow_dev, (std::size_t)n_overflow * sizeof(LbaExtent));
    if (cerr != cudaSuccess) {
        std::fprintf(stderr,
            "[nvme_storage] build_handle_template_: cudaMalloc(overflow, "
            "%u extents): %s\n", n_overflow, cudaGetErrorString(cerr));
        return false;
    }
    void* staging = nullptr;
    cerr = cudaMallocHost(&staging, (std::size_t)n_overflow * sizeof(LbaExtent));
    if (cerr != cudaSuccess) {
        std::fprintf(stderr,
            "[nvme_storage] build_handle_template_: cudaMallocHost("
            "overflow staging): %s\n", cudaGetErrorString(cerr));
        cudaFree(overflow_dev);
        return false;
    }
    std::memcpy(staging, file->extents.data() + kNvmeFileDeviceHandleInlineExtents,
               (std::size_t)n_overflow * sizeof(LbaExtent));
    cerr = cudaMemcpyAsync(overflow_dev, staging,
                           (std::size_t)n_overflow * sizeof(LbaExtent),
                           cudaMemcpyHostToDevice, stream);
    if (cerr != cudaSuccess) {
        std::fprintf(stderr,
            "[nvme_storage] build_handle_template_: cudaMemcpyAsync("
            "overflow): %s\n", cudaGetErrorString(cerr));
        cudaFreeHost(staging);
        cudaFree(overflow_dev);
        return false;
    }
    auto* sctx = new OverflowStagingFreeCtx{staging};
    cerr = cudaLaunchHostFunc(stream, &free_overflow_staging, sctx);
    if (cerr != cudaSuccess) {
        std::fprintf(stderr,
            "[nvme_storage] build_handle_template_: cudaLaunchHostFunc "
            "(free overflow staging) failed: %s (leaking staging buffer)\n",
            cudaGetErrorString(cerr));
        delete sctx;
    }

    out->extents_overflow = (LbaExtent*)overflow_dev;
    {
        std::lock_guard<std::mutex> lock(overflow_mtx_);
        // Bug-1 fix (R12 A-1): L2 LRU eviction removes the key from
        // TieredHandleCache's index but leaves the old overflow buffer
        // in overflow_by_file_.  When the same file is accessed again
        // via the COLD path, this builder runs again and would
        // overwrite the old pointer -- leaking the old GPU buffer.
        // Free it now (stream-ordered, same pattern as
        // erase_from_metadata_cache_) before overwriting.
        auto it = overflow_by_file_.find(file->id);
        if (it != overflow_by_file_.end() && it->second != nullptr) {
            auto* octx = new OverflowDeviceFreeCtx{it->second};
            cudaError_t free_cerr = cudaLaunchHostFunc(stream, &free_overflow_device, octx);
            if (free_cerr != cudaSuccess) {
                std::fprintf(stderr,
                    "[nvme_storage] build_handle_template_: failed to "
                    "schedule free of stale overflow buffer (file_id=%u): "
                    "%s (leaking to avoid use-after-free)\n",
                    (unsigned)file->id, cudaGetErrorString(free_cerr));
                delete octx;
            }
        }
        overflow_by_file_[file->id] = overflow_dev;
    }
    return true;
}

NvmeFileDeviceHandle* HostFsBackedNvmeStorage::acquire_device_handle(
    NvmeFile* file, GpuStreamHandle stream_h)
{
    if (file == nullptr || file->device == nullptr) return nullptr;
    auto* bp = static_cast<LocalNvmeDevice*>(file->device->backend_private);
    if (bp == nullptr || !bp->queue_group) return nullptr;
    const int cuda_device = (int)bp->queue_group->cuda_device();
    if (!check_current_device_("acquire_device_handle", cuda_device)) return nullptr;

    auto* cache = get_or_init_metadata_cache_(cuda_device);
    if (cache == nullptr) return nullptr;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_h);
    return cache->get_or_build(
        file->id,
        [this, file, stream](const NvmeFileId&, NvmeFileDeviceHandle* out) -> bool {
            return build_handle_template_(file, stream, out);
        },
        stream);
}

void HostFsBackedNvmeStorage::release_device_handle(NvmeFile* file, GpuStreamHandle stream_h)
{
    if (file == nullptr || file->device == nullptr) return;
    auto* bp = static_cast<LocalNvmeDevice*>(file->device->backend_private);
    if (bp == nullptr || !bp->queue_group) return;
    const int cuda_device = (int)bp->queue_group->cuda_device();
    if (!check_current_device_("release_device_handle", cuda_device)) return;

    std::unique_lock<std::mutex> lock(metadata_caches_mtx_);
    auto it = metadata_caches_.find(cuda_device);
    if (it == metadata_caches_.end()) return;
    auto* cache = it->second.get();
    lock.unlock();

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_h);
    cache->evict_from_l1(file->id, stream);
}

bool HostFsBackedNvmeStorage::acquire_device_handles_batch(
    NvmeFile* const* files, uint32_t count, GpuStreamHandle stream_h,
    NvmeFileDeviceHandle** out_handles)
{
    if (count == 0) return true;
    if (files == nullptr || files[0] == nullptr || files[0]->device == nullptr) return false;

    auto* bp = static_cast<LocalNvmeDevice*>(files[0]->device->backend_private);
    if (bp == nullptr || !bp->queue_group) return false;
    const int cuda_device = (int)bp->queue_group->cuda_device();
    if (!check_current_device_("acquire_device_handles_batch", cuda_device)) return false;

    auto* cache = get_or_init_metadata_cache_(cuda_device);
    if (cache == nullptr) return false;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_h);

    // The builder callback only receives a NvmeFileId key (not an
    // index into `files`), so build a key -> NvmeFile* lookup once
    // up front.  Only consulted for COLD misses, not every key.  Also
    // verify every file in the batch actually belongs to the SAME
    // cuda_device as files[0] -- this pool/cache is per-cuda_device,
    // so silently trusting only files[0] would mis-route any file
    // that happens to live on a different GPU into the wrong cache.
    std::vector<NvmeFileId> keys(count);
    std::unordered_map<NvmeFileId, NvmeFile*> file_by_key;
    file_by_key.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        if (files[i] == nullptr || files[i]->device == nullptr) return false;
        auto* bpi = static_cast<LocalNvmeDevice*>(files[i]->device->backend_private);
        if (bpi == nullptr || !bpi->queue_group) return false;
        const int di = (int)bpi->queue_group->cuda_device();
        if (di != cuda_device) {
            std::fprintf(stderr,
                "[nvme_storage] acquire_device_handles_batch: files[%u]='%s' "
                "cuda_device=%d != files[0]'s cuda_device=%d -- a single "
                "batch call must target ONE cuda_device (split the batch by "
                "device)\n", i, files[i]->name.c_str(), di, cuda_device);
            return false;
        }
        keys[i] = files[i]->id;
        file_by_key[files[i]->id] = files[i];
    }

    return cache->get_or_build_batch(
        keys.data(), count,
        [this, &file_by_key, stream](const NvmeFileId& key, NvmeFileDeviceHandle* out) -> bool {
            auto it = file_by_key.find(key);
            if (it == file_by_key.end()) {
                std::fprintf(stderr,
                    "[nvme_storage] acquire_device_handles_batch: builder "
                    "called with unknown key=%llu\n",
                    (unsigned long long)key);
                return false;
            }
            return build_handle_template_(it->second, stream, out);
        },
        stream, out_handles);
}

void HostFsBackedNvmeStorage::admit_to_cache(NvmeFile* file) {
    // R11.5: public wrapper around the private admit.  open_file no
    // longer calls this automatically (lazy); callers who want to
    // pre-warm the L2 cache call this explicitly.
    admit_to_metadata_cache_(file);
}

INvmeStorage::CacheStats HostFsBackedNvmeStorage::cache_stats() const {
    CacheStats agg{};
    std::lock_guard<std::mutex> lock(metadata_caches_mtx_);
    for (const auto& [cuda_dev, cache] : metadata_caches_) {
        (void)cuda_dev;
        auto s = cache->stats();
        agg.cold_builds   += s.cold_builds;
        agg.l1_hits       += s.l1_hits;
        agg.l2_hits       += s.l2_hits;
        agg.l1_promotions += s.l1_promotions;
        agg.l1_evictions  += s.l1_evictions;
        agg.l2_evictions  += s.l2_evictions;
    }
    return agg;
}

void HostFsBackedNvmeStorage::admit_to_metadata_cache_(NvmeFile* file)
{
    if (file == nullptr || file->device == nullptr) return;
    auto* bp = static_cast<LocalNvmeDevice*>(file->device->backend_private);
    if (bp == nullptr || !bp->queue_group) return;
    const int cuda_device = (int)bp->queue_group->cuda_device();

    // get_or_init_metadata_cache_ returns nullptr (cheaply, no CUDA call) when
    // the handle pool was never configured -- a pure host-side user
    // doing no GPU IO, for whom the cache is irrelevant.  In that case
    // admit is a no-op and open_file stays a plain host-fs operation.
    auto* cache = get_or_init_metadata_cache_(cuda_device);
    if (cache == nullptr) return;

    // Set the file's cuda_device current around the admit: the (rare)
    // overflow-buffer cudaMalloc inside build_handle_template_ must
    // land on the right device.  Restore afterwards so open_file stays
    // callable from any device context (it's otherwise a host-fs op).
    int prev_dev = -1;
    cudaError_t cerr = cudaGetDevice(&prev_dev);
    if (cerr != cudaSuccess) {
        std::fprintf(stderr,
            "[nvme_storage] admit_to_metadata_cache_: cudaGetDevice: %s\n",
            cudaGetErrorString(cerr));
        return;
    }
    if (prev_dev != cuda_device) cudaSetDevice(cuda_device);

    // Default stream: admit builds into L2 (host-pinned) only; the
    // sole GPU work it can trigger is the rare overflow-extent fill,
    // a one-time setup copy fine to order on the default stream.
    cudaStream_t stream = nullptr;
    cache->admit(
        file->id,
        [this, file, stream](const NvmeFileId&, NvmeFileDeviceHandle* out) -> bool {
            return build_handle_template_(file, stream, out);
        });

    if (prev_dev != cuda_device) cudaSetDevice(prev_dev);
}

void HostFsBackedNvmeStorage::erase_from_metadata_cache_(NvmeFile* file)
{
    if (file == nullptr || file->device == nullptr) return;
    auto* bp = static_cast<LocalNvmeDevice*>(file->device->backend_private);
    if (bp == nullptr || !bp->queue_group) return;
    const int cuda_device = (int)bp->queue_group->cuda_device();
    if (!check_current_device_("erase_from_metadata_cache_", cuda_device)) return;

    TieredHandleCache<NvmeFileDeviceHandle, NvmeFileId>* cache = nullptr;
    {
        std::lock_guard<std::mutex> lock(metadata_caches_mtx_);
        auto it = metadata_caches_.find(cuda_device);
        if (it != metadata_caches_.end()) cache = it->second.get();
    }
    if (cache == nullptr) return;   // never acquired -- nothing to erase

    cudaStream_t stream = nullptr;   // default stream; see the .h doc note
    cache->erase(file->id, stream);

    // Overflow buffer (if any) outlives L1/L2 residency -- free it
    // now, deferred via the same stream-ordered host-callback pattern
    // used elsewhere in this file so it never races a kernel that
    // might still be reading through extents_overflow.
    void* overflow_dev = nullptr;
    {
        std::lock_guard<std::mutex> lock(overflow_mtx_);
        auto it = overflow_by_file_.find(file->id);
        if (it != overflow_by_file_.end()) {
            overflow_dev = it->second;
            overflow_by_file_.erase(it);
        }
    }
    if (overflow_dev != nullptr) {
        auto* octx = new OverflowDeviceFreeCtx{overflow_dev};
        cudaError_t cerr = cudaLaunchHostFunc(stream, &free_overflow_device, octx);
        if (cerr != cudaSuccess) {
            std::fprintf(stderr,
                "[nvme_storage] erase_from_metadata_cache_: cudaLaunchHostFunc "
                "(free overflow device buf) failed: %s (leaking overflow "
                "buffer to avoid a potential use-after-free)\n",
                cudaGetErrorString(cerr));
            delete octx;
        }
    }
}

} // namespace tutti
