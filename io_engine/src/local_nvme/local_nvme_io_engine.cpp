/**
 * local_nvme_io_engine.cpp -- LocalNvmeIoEngine impl (R8.3).
 *
 * Glue between the R8.1 host_batch_builder and the R8.2
 * launch_nvme_batch_xfer GPU launch wrapper, behind the IIoEngine
 * interface.
 */

#include "local_nvme_io_engine.h"

#include "host_batch_builder.h"
#include "launch_batch.h"
#include "memory_subsystem.h"   // IMemorySubsystem (R7)

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace tutti {

LocalNvmeIoEngine::LocalNvmeIoEngine(IMemorySubsystem* mem,
                                     uint32_t          max_entries_per_batch)
    : mem_(mem),
      max_entries_(max_entries_per_batch),
      h_scratch_(max_entries_per_batch)
{
    if (mem_ == nullptr || max_entries_ == 0) {
        std::fprintf(stderr,
            "[io_engine] LocalNvmeIoEngine: bad input "
            "(mem=%p max_entries=%u)\n", (void*)mem_, max_entries_);
        std::abort();
    }
    const std::size_t bytes =
        sizeof(NvmeBatchEntry) * (std::size_t)max_entries_;
    cudaError_t cerr = cudaMalloc(&d_scratch_, bytes);
    if (cerr != cudaSuccess) {
        std::fprintf(stderr,
            "[io_engine] LocalNvmeIoEngine: cudaMalloc(%zu B) "
            "for d_scratch failed: %s\n",
            bytes, cudaGetErrorString(cerr));
        std::abort();
    }
}

LocalNvmeIoEngine::~LocalNvmeIoEngine() {
    if (d_scratch_ != nullptr) {
        cudaError_t cerr = cudaFree(d_scratch_);
        if (cerr != cudaSuccess) {
            // Best-effort -- can't throw from a destructor; emit a
            // diagnostic and move on.  Most likely cause: CUDA
            // context teardown already happened.
            std::fprintf(stderr,
                "[io_engine] ~LocalNvmeIoEngine: cudaFree(d_scratch) "
                "failed: %s (continuing teardown)\n",
                cudaGetErrorString(cerr));
        }
        d_scratch_ = nullptr;
    }
}

bool LocalNvmeIoEngine::launch_async_locked(
    const std::vector<NvmeBatchInputTensor>& inputs,
    bool                                     is_read,
    cudaStream_t                             stream)
{
    uint32_t count = 0;
    if (!build_nvme_batch(mem_, inputs, is_read,
                          h_scratch_.data(),
                          (uint32_t)h_scratch_.size(),
                          &count)) {
        std::fprintf(stderr,
            "[io_engine] LocalNvmeIoEngine: build_nvme_batch failed "
            "(inputs=%zu is_read=%d max_entries=%u)\n",
            inputs.size(), (int)is_read, max_entries_);
        return false;
    }
    if (count == 0) {
        std::fprintf(stderr,
            "[io_engine] LocalNvmeIoEngine: build produced 0 entries\n");
        return false;
    }

    // C-1/C-2: Promote this batch's cache-managed PRP-list pages into the
    // GPU-DMA L1 tier and patch their descriptors' prp2 to the current
    // L1 IOVAs -- ON `stream`, BEFORE the H2D of the entry array and the
    // kernel launch below, so the NVMe controller fetches valid PRP
    // lists.  No-op for tensors that need no PRP list or use the owned
    // always-resident fallback.  The prp2 patch is stream-ordered ahead
    // of the kernel (same stream), so no extra fence is needed here.
    {
        std::vector<MemoryRegion*> regions;
        regions.reserve(inputs.size());
        for (const auto& in : inputs) regions.push_back(in.tensor_region);
        if (!mem_->ensure_prp_pages_resident(regions, stream)) {
            std::fprintf(stderr,
                "[io_engine] LocalNvmeIoEngine: ensure_prp_pages_resident "
                "failed (inputs=%zu)\n", inputs.size());
            return false;
        }
    }

    cudaError_t cerr = cudaMemcpyAsync(
        d_scratch_, h_scratch_.data(),
        sizeof(NvmeBatchEntry) * (std::size_t)count,
        cudaMemcpyHostToDevice, stream);
    if (cerr != cudaSuccess) {
        std::fprintf(stderr,
            "[io_engine] LocalNvmeIoEngine: cudaMemcpyAsync H2D "
            "(%u entries) failed: %s\n",
            count, cudaGetErrorString(cerr));
        return false;
    }
    cerr = launch_nvme_batch_xfer(stream, d_scratch_, count, is_read,
                                   /*threads_per_block=*/32);
    if (cerr != cudaSuccess) {
        std::fprintf(stderr,
            "[io_engine] LocalNvmeIoEngine: launch_nvme_batch_xfer "
            "(%u entries, is_read=%d) failed: %s\n",
            count, (int)is_read, cudaGetErrorString(cerr));
        return false;
    }
    return true;
}

bool LocalNvmeIoEngine::submit_batch_async(
    const std::vector<NvmeBatchInputTensor>& inputs,
    bool                                     is_read,
    cudaStream_t                             stream)
{
    return launch_async_locked(inputs, is_read, stream);
}

bool LocalNvmeIoEngine::submit_batch(
    const std::vector<NvmeBatchInputTensor>& inputs,
    bool                                     is_read,
    cudaStream_t                             stream)
{
    if (!launch_async_locked(inputs, is_read, stream)) {
        return false;
    }
    cudaError_t cerr = cudaStreamSynchronize(stream);
    if (cerr != cudaSuccess) {
        std::fprintf(stderr,
            "[io_engine] LocalNvmeIoEngine: cudaStreamSynchronize "
            "failed: %s\n", cudaGetErrorString(cerr));
        return false;
    }
    cerr = cudaGetLastError();
    if (cerr != cudaSuccess) {
        std::fprintf(stderr,
            "[io_engine] LocalNvmeIoEngine: kernel reported error "
            "after sync: %s\n", cudaGetErrorString(cerr));
        return false;
    }
    return true;
}

} // namespace tutti
