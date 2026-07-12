/**
 * kv_cache_io_adapter.cpp -- KvCacheIoAdapter implementation.
 *
 * Pure host code: builds NvmeBatchInputTensor lists with the legacy
 * per-layer K/V byte offsets, auto-chunks them under the io_engine's
 * scratch capacity, and submits each chunk through the Coordinator.
 *
 * Deliberately libnvm-free: it includes only the Coordinator + io_engine
 * surfaces and treats MemoryRegion* / GpuFileHandle* as opaque handles.
 * The per-region entry count needed for chunking comes from
 * Coordinator::batch_entry_count(), which keeps the memory subsystem's
 * <nvm_types.h>-bearing headers inside coordinator.cu (a .cu TU).  This
 * is what lets the adapter stay a plain CXX translation unit.
 */

#include "kv_cache_io_adapter.h"

#include "coordinator.h"               // Coordinator (+ NvmeBatchInputTensor)
#include "io_engine.h"                 // IIoEngine::max_entries_per_batch

#include <cstdio>
#include <vector>

namespace tutti {

namespace {

// Greedily pack `inputs` into chunks whose flattened entry count stays
// <= max_entries, submitting each chunk blocking on `stream`.  This is
// the D2 fix: the legacy 512-entry BatchIoPool loop, expressed against
// the single-scratch v0.1 engine.
bool submit_chunked(Coordinator&                             coord,
                    uint32_t                                 max_entries,
                    const std::vector<NvmeBatchInputTensor>& inputs,
                    bool                                     is_read,
                    cudaStream_t                             stream) {
    if (inputs.empty()) {
        std::fprintf(stderr, "[kv_adapter] submit_chunked: empty inputs\n");
        return false;
    }
    if (max_entries == 0) {
        std::fprintf(stderr,
            "[kv_adapter] submit_chunked: max_entries==0 (Coordinator "
            "booted?)\n");
        return false;
    }

    std::size_t i = 0;
    while (i < inputs.size()) {
        std::vector<NvmeBatchInputTensor> chunk;
        uint32_t acc = 0;
        while (i < inputs.size()) {
            const uint32_t n = coord.batch_entry_count(inputs[i].tensor_region);
            if (n == 0) {
                std::fprintf(stderr,
                    "[kv_adapter] input %zu produced 0 entries "
                    "(register_tensor with granularity > 0?)\n", i);
                return false;
            }
            if (n > max_entries) {
                std::fprintf(stderr,
                    "[kv_adapter] input %zu needs %u entries > engine "
                    "max_entries_per_batch=%u: a single tensor cannot be "
                    "split across batches in v0.1 (raise "
                    "max_entries_per_batch)\n", i, n, max_entries);
                return false;
            }
            if (acc + n > max_entries && !chunk.empty()) break;  // flush
            chunk.push_back(inputs[i]);
            acc += n;
            ++i;
        }
        if (!coord.submit_batch(chunk, is_read, stream)) {
            std::fprintf(stderr,
                "[kv_adapter] submit_batch(%zu tensors, %u entries, "
                "is_read=%d) failed\n", chunk.size(), acc, (int)is_read);
            return false;
        }
    }
    return true;
}

}  // namespace

KvCacheIoAdapter::KvCacheIoAdapter(Coordinator& coord,
                                   uint64_t     tensor_size,
                                   bool         use_mla)
    : coord_(coord),
      tensor_size_(tensor_size),
      use_mla_(use_mla),
      max_entries_(0) {
    IIoEngine* eng = coord_.io_engine();
    if (eng != nullptr) max_entries_ = eng->max_entries_per_batch();
    if (max_entries_ == 0) {
        std::fprintf(stderr,
            "[kv_adapter] WARNING: io_engine unavailable or "
            "max_entries_per_batch==0; submits will fail until the "
            "Coordinator is booted.\n");
    }
}

bool KvCacheIoAdapter::xfer(int                                 layer_idx,
                            const std::vector<MemoryRegion*>&   k_regions,
                            const std::vector<MemoryRegion*>&   v_regions,
                            const std::vector<GpuFileHandle*>&  file_handles,
                            bool                                is_read,
                            cudaStream_t                        stream) {
    if (k_regions.empty() ||
        k_regions.size() != v_regions.size() ||
        k_regions.size() != file_handles.size()) {
        std::fprintf(stderr,
            "[kv_adapter] xfer: size mismatch k=%zu v=%zu files=%zu\n",
            k_regions.size(), v_regions.size(), file_handles.size());
        return false;
    }
    if (layer_idx < 0) {
        std::fprintf(stderr, "[kv_adapter] xfer: negative layer_idx=%d\n",
                     layer_idx);
        return false;
    }

    // Legacy geminifs_batched_xfer offsets:
    //   K @ layer * len * 2 ,  V @ layer * len * 2 + len .
    const uint64_t base  = (uint64_t)layer_idx * tensor_size_ * 2ull;
    const uint64_t k_off = base;
    const uint64_t v_off = base + tensor_size_;

    std::vector<NvmeBatchInputTensor> inputs;
    inputs.reserve(k_regions.size() * 2);
    for (std::size_t b = 0; b < k_regions.size(); ++b) {
        if (k_regions[b] == nullptr || v_regions[b] == nullptr ||
            file_handles[b] == nullptr) {
            std::fprintf(stderr,
                "[kv_adapter] xfer: null at block %zu "
                "(k=%p v=%p file=%p)\n", b, (void*)k_regions[b],
                (void*)v_regions[b], (void*)file_handles[b]);
            return false;
        }
        inputs.push_back(NvmeBatchInputTensor{k_regions[b], file_handles[b], k_off});
        inputs.push_back(NvmeBatchInputTensor{v_regions[b], file_handles[b], v_off});
    }

    return submit_chunked(coord_, max_entries_, inputs, is_read, stream);
}

bool KvCacheIoAdapter::xfer_unified(int                                 layer_idx,
                                    const std::vector<MemoryRegion*>&   regions,
                                    const std::vector<GpuFileHandle*>&  file_handles,
                                    bool                                is_read,
                                    cudaStream_t                        stream) {
    if (regions.empty() || regions.size() != file_handles.size()) {
        std::fprintf(stderr,
            "[kv_adapter] xfer_unified: size mismatch r=%zu files=%zu\n",
            regions.size(), file_handles.size());
        return false;
    }
    if (layer_idx < 0) {
        std::fprintf(stderr,
            "[kv_adapter] xfer_unified: negative layer_idx=%d\n", layer_idx);
        return false;
    }

    // MLA: single tensor per layer -> offset layer * len .
    const uint64_t off = (uint64_t)layer_idx * tensor_size_;

    std::vector<NvmeBatchInputTensor> inputs;
    inputs.reserve(regions.size());
    for (std::size_t b = 0; b < regions.size(); ++b) {
        if (regions[b] == nullptr || file_handles[b] == nullptr) {
            std::fprintf(stderr,
                "[kv_adapter] xfer_unified: null at block %zu "
                "(r=%p file=%p)\n", b, (void*)regions[b],
                (void*)file_handles[b]);
            return false;
        }
        inputs.push_back(NvmeBatchInputTensor{regions[b], file_handles[b], off});
    }

    return submit_chunked(coord_, max_entries_, inputs, is_read, stream);
}

bool KvCacheIoAdapter::resolve_handles(const std::vector<GpuFileId>&  file_ids,
                                       std::vector<GpuFileHandle*>&   out_handles,
                                       cudaStream_t                    stream) {
    // Pre-validate every id BEFORE the capacity check / any GPU work
    // -- handle_for_batch would also catch an unknown id, but only at
    // the first one it encounters and only after this adapter has
    // already committed to the batch; checking here gives a precise,
    // per-id error up front (queries the same id -> GpuFile*
    // directory handle_for_batch uses internally).
    for (std::size_t i = 0; i < file_ids.size(); ++i) {
        if (!coord_.has_gpu_file(file_ids[i])) {
            std::fprintf(stderr,
                "[kv_adapter] resolve_handles: file_ids[%zu]=%u not "
                "created/opened this session (call open_gpu_file first)\n",
                i, file_ids[i]);
            return false;
        }
    }


    // R11.3: ONE flattened acquire across every file's shards (see
    // Coordinator::handle_for_batch / block_storage.h's
    // acquire_device_handles_batch doc) instead of `file_ids.size()`
    // separate ones.  Returned pointers' contents are only guaranteed
    // valid for GPU work queued on `stream` AFTER this call;
    // xfer()/xfer_unified() below submit on the SAME `stream`, so
    // ordinary CUDA stream-ordering gives correctness with no
    // CPU-side wait here.
    out_handles.assign(file_ids.size(), nullptr);
    if (!coord_.handle_for_batch(file_ids.data(), (uint32_t)file_ids.size(),
                                stream, out_handles.data())) {
        std::fprintf(stderr,
            "[kv_adapter] resolve_handles: handle_for_batch(%zu ids) "
            "failed\n", file_ids.size());
        return false;
    }
    return true;
}

// ---- GpuFileId flavours: resolve through the handle cache, delegate. ----

bool KvCacheIoAdapter::batched_write(int                                 layer_idx,
                                     const std::vector<MemoryRegion*>&   k_regions,
                                     const std::vector<MemoryRegion*>&   v_regions,
                                     const std::vector<GpuFileId>&       file_ids,
                                     cudaStream_t                        stream) {
    std::vector<GpuFileHandle*> handles;
    if (!resolve_handles(file_ids, handles, stream)) return false;
    return xfer(layer_idx, k_regions, v_regions, handles, false, stream);
}

bool KvCacheIoAdapter::batched_read(int                                 layer_idx,
                                    const std::vector<MemoryRegion*>&   k_regions,
                                    const std::vector<MemoryRegion*>&   v_regions,
                                    const std::vector<GpuFileId>&       file_ids,
                                    cudaStream_t                        stream) {
    std::vector<GpuFileHandle*> handles;
    if (!resolve_handles(file_ids, handles, stream)) return false;
    return xfer(layer_idx, k_regions, v_regions, handles, true, stream);
}

bool KvCacheIoAdapter::batched_write_unified(int                                 layer_idx,
                                             const std::vector<MemoryRegion*>&   regions,
                                             const std::vector<GpuFileId>&       file_ids,
                                             cudaStream_t                        stream) {
    std::vector<GpuFileHandle*> handles;
    if (!resolve_handles(file_ids, handles, stream)) return false;
    return xfer_unified(layer_idx, regions, handles, false, stream);
}

bool KvCacheIoAdapter::batched_read_unified(int                                 layer_idx,
                                            const std::vector<MemoryRegion*>&   regions,
                                            const std::vector<GpuFileId>&       file_ids,
                                            cudaStream_t                        stream) {
    std::vector<GpuFileHandle*> handles;
    if (!resolve_handles(file_ids, handles, stream)) return false;
    return xfer_unified(layer_idx, regions, handles, true, stream);
}

// ---- GpuFileHandle* flavours: low-level, caller-owned handles. ----

bool KvCacheIoAdapter::batched_write(int                                 layer_idx,
                                     const std::vector<MemoryRegion*>&   k_regions,
                                     const std::vector<MemoryRegion*>&   v_regions,
                                     const std::vector<GpuFileHandle*>&  file_handles,
                                     cudaStream_t                        stream) {
    return xfer(layer_idx, k_regions, v_regions, file_handles, false, stream);
}

bool KvCacheIoAdapter::batched_read(int                                 layer_idx,
                                    const std::vector<MemoryRegion*>&   k_regions,
                                    const std::vector<MemoryRegion*>&   v_regions,
                                    const std::vector<GpuFileHandle*>&  file_handles,
                                    cudaStream_t                        stream) {
    return xfer(layer_idx, k_regions, v_regions, file_handles, true, stream);
}

bool KvCacheIoAdapter::batched_write_unified(int                                 layer_idx,
                                             const std::vector<MemoryRegion*>&   regions,
                                             const std::vector<GpuFileHandle*>&  file_handles,
                                             cudaStream_t                        stream) {
    return xfer_unified(layer_idx, regions, file_handles, false, stream);
}

bool KvCacheIoAdapter::batched_read_unified(int                                 layer_idx,
                                            const std::vector<MemoryRegion*>&   regions,
                                            const std::vector<GpuFileHandle*>&  file_handles,
                                            cudaStream_t                        stream) {
    return xfer_unified(layer_idx, regions, file_handles, true, stream);
}

} // namespace tutti
