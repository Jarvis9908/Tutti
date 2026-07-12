#ifndef __TUTTI_IO_ENGINE_LOCAL_NVME_HOST_BATCH_BUILDER_H__
#define __TUTTI_IO_ENGINE_LOCAL_NVME_HOST_BATCH_BUILDER_H__

/**
 * host_batch_builder.h -- flatten (tensor x sub-slice) into NvmeBatchEntry[].
 *
 * Layer: io_engine, local_nvme backend.  Pure host-side helper.
 *
 * Spec: doc/refactor/R8_io_engine_plan.md §3.4.
 *
 * Mirrors the legacy monolithic batched-xfer host loop
 * (fill_ctx + outer flatten) from the pre-refactor GPU-file path.
 *
 * What it does NOT do:
 *   - allocate the output buffer (caller passes pre-sized array)
 *   - copy the result to GPU (caller cudaMemcpyAsync's after build)
 *   - launch any kernel (R8.2 lands the kernel + the cudaMemcpy
 *     orchestration on top of this builder).
 *
 * That keeps R8.1 a pure CPU correctness step: smoke tests can
 * field-by-field compare the output against the legacy formula
 * without involving any CUDA stream, kernel, or PRP submission.
 */

#include <cstdint>
#include <vector>

namespace tutti {

class IMemorySubsystem;
struct NvmeBatchEntry;
struct NvmeBatchInputTensor;

/// Build a flat NvmeBatchEntry array covering every (tensor, sub-slice)
/// across the given inputs.
///
/// Output ordering:
///   for each input T (in input order):
///     for each IoSliceView s of T's tensor_region (in slice_addr order):
///       for each sub-slice k in [0, s.num_ios):
///         emit one entry, with prp_idx == GLOBAL sub-slice index
///         within the tensor (i.e. running counter, NOT k).
///
/// Why global prp_idx (not slice-local k):
///   Kernel computes gpu_blk = file_offset/blk_size + prp_idx and
///   then `fd_idx = gpu_blk % num_shards`.  For the formula to map
///   to one stream of round-robin striped IOs across the tensor,
///   prp_idx MUST be the tensor-global running index.  Using
///   slice-local k would only work coincidentally when the tensor
///   has exactly one IoSliceView (granularity == tensor_size).
///   See R8_io_engine_plan.md §3.1 step 3a.
///
/// @param mem            R7 memory subsystem; needed for list_io_slices().
/// @param inputs         Input tensors with their GpuFile shard handles
///                       and per-tensor base byte offsets.
/// @param is_read        Direction; copied into every emitted entry's
///                       is_read field (per-entry direction is reserved
///                       for future use; R8.2 kernel reads the batch-
///                       uniform direction parameter).
/// @param out_entries    Caller-owned host-side array of length
///                       `out_capacity`.
/// @param out_capacity   Capacity of out_entries[].
/// @param out_count      On success, set to the number of entries written.
///                       On failure (insufficient capacity, missing
///                       IoSliceView for an input), set to the count
///                       written before the failure (caller should treat
///                       the partial output as undefined).
///
/// @return true iff every input tensor was fully written and total
///         entry count <= out_capacity.  false otherwise (with stderr
///         diagnostic).
bool build_nvme_batch(IMemorySubsystem*                          mem,
                      const std::vector<NvmeBatchInputTensor>&   inputs,
                      bool                                       is_read,
                      NvmeBatchEntry*                            out_entries,
                      uint32_t                                   out_capacity,
                      uint32_t*                                  out_count);

} // namespace tutti

#endif // __TUTTI_IO_ENGINE_LOCAL_NVME_HOST_BATCH_BUILDER_H__
