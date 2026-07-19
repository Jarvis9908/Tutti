#ifndef __TUTTI_IO_ENGINE_IO_ENGINE_H__
#define __TUTTI_IO_ENGINE_IO_ENGINE_H__

/**
 * io_engine.h -- host-side IO submission SPI.
 *
 * Layer: io_engine (top-level interface).  Built on device_manager /
 * block_storage / memory plus the host_batch_builder +
 * nvme_batch_xfer_kernel data plane.
 *
 * Role:
 *   IIoEngine is the single host-side entry point for callers that
 *   want to submit a batch of NVMe IOs against memory-registered
 *   tensors and block_storage GpuFile shards.  The runtime / examples
 *   / Coordinator hold an IIoEngine* and never touch
 *   build_nvme_batch / launch_nvme_batch_xfer directly.
 *
 *   Lower-level callers that need to manage their own scratch buffer
 *   or stream-ordering (smokes, micro-benchmarks) can still go
 *   straight to the underlying free functions; IIoEngine is the
 *   recommended path, not the only one.
 *
 * Concurrency contract (v0.1 LocalNvmeIoEngine):
 *   One engine instance owns ONE GPU-resident scratch buffer for
 *   NvmeBatchEntries.  Submits from different threads MUST serialize
 *   themselves (e.g. one engine per stream / per worker) or the
 *   scratch buffer races.  Multiple consecutive submits on the SAME
 *   stream are safe -- the cudaMemcpyAsync into the scratch is
 *   stream-ordered, so the next submit's H2D won't fire until the
 *   previous kernel has consumed the staged entries.
 *
 * Direction model:
 *   Each batch carries one direction (`is_read`).  Mixed-direction
 *   batches are NOT supported in v0.1; NvmeBatchEntry has a per-entry
 *   is_read field reserved for that future case.
 */

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "local_nvme/nvme_batch.h"   // NvmeBatchInputTensor

namespace tutti {

class IIoEngine {
public:
    virtual ~IIoEngine() = default;

    /// Submit one batch and block until it completes.
    ///
    /// Internally:
    ///   1. host-side `build_nvme_batch` flattens
    ///      (input tensor x sub-slice) -> NvmeBatchEntry[]
    ///   2. `cudaMemcpyAsync` host_scratch -> d_scratch on `stream`
    ///   3. `launch_nvme_batch_xfer` on `stream`
    ///   4. `cudaStreamSynchronize(stream)`
    ///   5. checks cudaGetLastError() to surface kernel-side failures
    ///
    /// `stream` MAY carry unrelated work; this call hard-syncs the
    /// whole stream as part of its semantics.  Pass a dedicated
    /// stream when that matters.
    ///
    /// @return true iff every step succeeded.  On any failure the
    ///         engine emits a `[io_engine]` diagnostic and the
    ///         caller should treat the device tensor / shards as
    ///         having undefined content for this batch.
    virtual bool submit_batch(
        const std::vector<NvmeBatchInputTensor>& inputs,
        bool                                     is_read,
        cudaStream_t                             stream) = 0;

    /// Async variant: returns after the kernel launch is queued on
    /// `stream`.  Caller is responsible for the stream-sync / event /
    /// callback that observes completion.
    ///
    /// @return true iff host-side build + launch succeeded.  Kernel-
    ///         side failures (CQ-level errors) are NOT detected by
    ///         this call -- they surface on the caller's eventual
    ///         stream-sync via cudaGetLastError().
    virtual bool submit_batch_async(
        const std::vector<NvmeBatchInputTensor>& inputs,
        bool                                     is_read,
        cudaStream_t                             stream) = 0;

    /// Capacity of the engine's internal NvmeBatchEntry scratch.
    /// `build_nvme_batch` will reject any input whose flattened
    /// entry count exceeds this.  Set at construction; immutable.
    virtual uint32_t max_entries_per_batch() const = 0;
};

} // namespace tutti

#endif // __TUTTI_IO_ENGINE_IO_ENGINE_H__
