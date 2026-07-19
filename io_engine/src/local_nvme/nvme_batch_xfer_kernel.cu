/**
 * nvme_batch_xfer_kernel.cu -- GPU batch IO kernel + launch wrapper.
 *
 * Types involved:
 *
 *   NvmeBatchEntry::shards + num_shards  -- per-IO shard pointer table
 *   NvmeBatchEntry::prp_entry            -- AddressDescriptor* (PRP)
 *   submit_{read,write}_one              -- on-GPU submit primitives
 *                                            (nvme_storage_device.cuh)
 *
 * Stripe selection runs on the GPU: the same input tensor's
 * sub-slices land on different shards in one batch via
 * `fd_idx = gpu_blk % num_shards`.
 *
 * Virtual-file -> physical-LBA translation runs inside
 * `submit_*_one` (resolve_lba walks the NvmeFile's extent list).
 */

#include "launch_batch.h"
#include "nvme_batch.h"

// AddressDescriptor (carries prp1/prp2/data_length).
#include "memory_subsystem.h"

// On-GPU submit primitives + NvmeFileDeviceHandle.
#include "nvme_storage_device.cuh"
#include "nvme_file_device_handle.h"

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

namespace tutti {

namespace {

__global__
void nvme_batch_xfer_kernel(const NvmeBatchEntry* entries,
                            uint32_t              count,
                            bool                  is_read)
{
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= count) return;

    // Value-copy 40 bytes from device memory.  This is what the
    // legacy `auto ctx = io_ctx->d_ioctxs[tid]` does too -- we
    // dereference the GPU-resident pointer fields below.
    const NvmeBatchEntry e = entries[tid];

    if (e.prp_entry == nullptr ||
        e.shards    == nullptr ||
        e.num_shards == 0) {
        // Defensive: legacy returns silently on a null prp_entry /
        // empty span.  Use a one-shot printf so a misbuild prints
        // exactly once per offending tid in dmesg-style logs.
        printf("nvme_batch_xfer_kernel: tid=%u skipped "
               "(prp_entry=%p shards=%p num_shards=%u)\n",
               tid, (const void*)e.prp_entry,
               (const void*)e.shards, e.num_shards);
        return;
    }

    // (3a) Stripe selection -- legacy lines 1644-1646:
    //   gpu_blk  = (file_offset / blk_size) + prp_idx
    //   fd_idx   = gpu_blk % nvme_file.size()
    //   file_off = (gpu_blk / nvme_file.size()) * blk_size
    const uint64_t blk_size = e.prp_entry->data_length;
    if (blk_size == 0) {
        printf("nvme_batch_xfer_kernel: tid=%u zero blk_size\n", tid);
        return;
    }
    const uint64_t gpu_blk  = (e.file_offset / blk_size)
                            + (uint64_t)e.prp_idx;
    const uint32_t fd_idx   = (uint32_t)(gpu_blk % e.num_shards);
    const uint64_t file_off = (gpu_blk / e.num_shards) * blk_size;

    NvmeFileDeviceHandle* dh = e.shards[fd_idx];
    if (dh == nullptr) {
        printf("nvme_batch_xfer_kernel: tid=%u shards[%u] == nullptr\n",
               tid, fd_idx);
        return;
    }

    // (3b) Virtual file offset -> physical NVMe LBA translation +
    //      command issue, both inside R5b's submit_*_one.
    if (is_read) {
        submit_read_one (dh,
                         e.prp_entry->prp1,
                         e.prp_entry->prp2,
                         file_off,
                         blk_size);
    } else {
        submit_write_one(dh,
                         e.prp_entry->prp1,
                         e.prp_entry->prp2,
                         file_off,
                         blk_size);
    }
}

} // anonymous namespace

cudaError_t launch_nvme_batch_xfer(cudaStream_t          stream,
                                   const NvmeBatchEntry* d_entries,
                                   uint32_t              count,
                                   bool                  is_read,
                                   uint32_t              threads_per_block)
{
    if (d_entries == nullptr) {
        std::fprintf(stderr,
            "[io_engine/launch_nvme_batch_xfer] d_entries == nullptr\n");
        return cudaErrorInvalidValue;
    }
    if (count == 0) {
        std::fprintf(stderr,
            "[io_engine/launch_nvme_batch_xfer] count == 0\n");
        return cudaErrorInvalidValue;
    }
    if (threads_per_block == 0) {
        std::fprintf(stderr,
            "[io_engine/launch_nvme_batch_xfer] threads_per_block == 0\n");
        return cudaErrorInvalidValue;
    }

    const uint32_t blocks =
        (count + threads_per_block - 1) / threads_per_block;

    nvme_batch_xfer_kernel
        <<<blocks, threads_per_block, 0, stream>>>(
            d_entries, count, is_read);

    cudaError_t cerr = cudaGetLastError();
    if (cerr != cudaSuccess) {
        std::fprintf(stderr,
            "[io_engine/launch_nvme_batch_xfer] kernel launch "
            "(grid=%u, block=%u, count=%u) failed: %s\n",
            blocks, threads_per_block, count,
            cudaGetErrorString(cerr));
    }
    return cerr;
}

} // namespace tutti
