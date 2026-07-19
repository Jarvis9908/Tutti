#ifndef __TUTTI_IO_ENGINE_LOCAL_NVME_LOCAL_NVME_IO_ENGINE_H__
#define __TUTTI_IO_ENGINE_LOCAL_NVME_LOCAL_NVME_IO_ENGINE_H__

/**
 * local_nvme_io_engine.h -- LocalNvmeIoEngine.
 *
 * Concrete IIoEngine for the local NVMe data plane.  Wraps the
 * host batch builder + GPU launch wrapper into one
 * lifetime-managed object.
 *
 * Owned state:
 *   - host_scratch  : std::vector<NvmeBatchEntry>(max_entries)
 *   - d_scratch     : cudaMalloc'd NvmeBatchEntry array of the
 *                     same length, alive for the engine's lifetime.
 *
 * Construction / teardown:
 *   - Construction validates inputs and aborts on cudaMalloc failure
 *     (callers can't recover from it sensibly).  Use a try/check
 *     factory pattern at the call site if soft-failure is needed.
 *   - Destruction cudaFree's d_scratch and clears host_scratch.
 */

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "../io_engine.h"
#include "nvme_batch.h"

namespace tutti {

class IMemorySubsystem;

class LocalNvmeIoEngine : public IIoEngine {
public:
    LocalNvmeIoEngine(IMemorySubsystem* mem,
                      uint32_t          max_entries_per_batch);
    ~LocalNvmeIoEngine() override;

    LocalNvmeIoEngine(const LocalNvmeIoEngine&)            = delete;
    LocalNvmeIoEngine& operator=(const LocalNvmeIoEngine&) = delete;

    bool submit_batch(
        const std::vector<NvmeBatchInputTensor>& inputs,
        bool                                     is_read,
        cudaStream_t                             stream) override;

    bool submit_batch_async(
        const std::vector<NvmeBatchInputTensor>& inputs,
        bool                                     is_read,
        cudaStream_t                             stream) override;

    uint32_t max_entries_per_batch() const override { return max_entries_; }

private:
    /// Internal: build host_scratch_, H2D async, kernel launch.
    /// Stops at any host-side error (does not stream-sync).
    bool launch_async_locked(
        const std::vector<NvmeBatchInputTensor>& inputs,
        bool                                     is_read,
        cudaStream_t                             stream);

    IMemorySubsystem*              mem_         = nullptr;
    uint32_t                       max_entries_ = 0;
    std::vector<NvmeBatchEntry>    h_scratch_;
    NvmeBatchEntry*                d_scratch_   = nullptr;
};

} // namespace tutti

#endif // __TUTTI_IO_ENGINE_LOCAL_NVME_LOCAL_NVME_IO_ENGINE_H__
