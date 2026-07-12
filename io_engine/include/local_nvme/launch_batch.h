#ifndef __TUTTI_IO_ENGINE_LOCAL_NVME_LAUNCH_BATCH_H__
#define __TUTTI_IO_ENGINE_LOCAL_NVME_LAUNCH_BATCH_H__

/**
 * launch_batch.h -- host-callable wrapper that launches the
 * nvme_batch_xfer_kernel for an already-staged device-resident
 * NvmeBatchEntry array (R8.2).
 *
 * Spec: doc/refactor/R8_io_engine_plan.md §4 (R8.2 milestone) +
 *       §3.3 (the kernel body).
 *
 * Why a separate header from `nvme_batch.h`:
 *   - `nvme_batch.h` carries POD types only; both host TUs and
 *     device TUs include it.
 *   - This header pulls in `<cuda_runtime.h>` (for cudaStream_t /
 *     cudaError_t) and is for host-side launch sites only.
 *
 * Memory contract:
 *   - `d_entries` MUST be GPU-resident (cudaMalloc'd or pinned-mapped
 *     on a device the launching CUDA context can reach).  The caller
 *     is responsible for cudaMemcpyAsync'ing the host-built batch
 *     into this buffer BEFORE invoking the launch.
 *   - The kernel reads `e.shards[fd_idx]` and `e.prp_entry->...`
 *     directly: those pointers (set by `build_nvme_batch`) MUST be
 *     GPU-resident.  In the standard pipeline:
 *       e.shards    = GpuFileHandle::d_shards_dev   (R8.1)
 *       e.prp_entry = IoSliceView::d_ios + sub      (R7)
 *     both already-GPU-resident, no extra staging needed.
 *
 * Lifetime:
 *   - The MemoryRegion holding `d_entries`, the GpuFileHandle whose
 *     `d_shards_dev` is referenced, and the MemoryRegion whose R7
 *     `d_ios` is referenced, MUST all stay alive until the launch's
 *     stream-ordered work has completed (caller stream sync /
 *     event / callback).
 */

#include <cstdint>
#include <cuda_runtime.h>

namespace tutti {

struct NvmeBatchEntry;

/// Launch `nvme_batch_xfer_kernel<<<grid, block, 0, stream>>>(...)`.
///
/// One CUDA thread per entry.  Each thread:
///   1. computes  gpu_blk  = e.file_offset/blk_size + e.prp_idx,
///                fd_idx   = gpu_blk % e.num_shards,
///                file_off = (gpu_blk / e.num_shards) * blk_size
///   2. resolves  dh = e.shards[fd_idx]
///   3. calls     submit_{read,write}_one(dh, prp1, prp2, file_off,
///                                        blk_size)
///
/// Direction is batch-uniform (`is_read`); the per-entry `is_read`
/// field on NvmeBatchEntry is reserved for future mixed-direction
/// batches and is currently ignored by the kernel.
///
/// @param stream             CUDA stream.  Pass 0 for the default stream.
/// @param d_entries          GPU-resident NvmeBatchEntry array.
/// @param count              Number of entries.
/// @param is_read            true -> read; false -> write.
/// @param threads_per_block  Block dim.  Default 32 to mirror legacy.
///
/// @return cudaSuccess on successful launch (kernel completion is
///         stream-ordered; this returns BEFORE the kernel finishes).
///         Any other code is a launch-time failure (count==0,
///         d_entries==nullptr, etc.); the caller should NOT
///         stream-sync after a non-success return.
cudaError_t launch_nvme_batch_xfer(cudaStream_t          stream,
                                   const NvmeBatchEntry* d_entries,
                                   uint32_t              count,
                                   bool                  is_read,
                                   uint32_t              threads_per_block = 32);

} // namespace tutti

#endif // __TUTTI_IO_ENGINE_LOCAL_NVME_LAUNCH_BATCH_H__
