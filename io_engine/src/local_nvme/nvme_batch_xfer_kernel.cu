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
 * Stripe selection runs on the GPU via block_storage's own
 * `gpu_file_resolve` (tensor_size granularity): one tensor lands
 * ENTIRELY on one shard; its MDTS fan-out sub-IOs are contiguous
 * within that shard (shard_base + prp_idx * sub_io).  Hosting the
 * shard choice on the GPU is what lets one batch mix tensors from
 * every shard without exploding the host-side entry count.
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

// block_storage's authoritative stripe translation (host+device).
#include "gpu_file_resolve.h"

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

    // (3a) Stripe selection -- via block_storage's gpu_file_resolve at
    // TENSOR_SIZE granularity (R6 invariant).  The previous form used
    // the sub-IO size (prp_entry->data_length, i.e. MDTS) as the
    // stripe unit; that is only equivalent when tensor_size == MDTS
    // (one sub-IO per tensor).  With tensor_size > MDTS the fan-out
    // sub-IOs were round-robined across shards at sub-IO granularity,
    // scattering one tensor's pieces onto the WRONG shards and into
    // wrong shard offsets (resolve_lba failures / cross K/V
    // corruption).  Correct form: the whole tensor lives on ONE shard;
    // sub-IOs are contiguous within it.
    const uint64_t sub_io = e.prp_entry->data_length;
    if (sub_io == 0) {
        printf("nvme_batch_xfer_kernel: tid=%u zero sub_io\n", tid);
        return;
    }
    if (e.tensor_size == 0 || (e.file_offset % e.tensor_size) != 0) {
        printf("nvme_batch_xfer_kernel: tid=%u bad tensor_size=%u "
               "file_offset=%llu\n", tid, e.tensor_size,
               (unsigned long long)e.file_offset);
        return;
    }
    uint32_t fd_idx   = 0;
    uint64_t base_off = 0;
    gpu_file_resolve(e.tensor_size, e.num_shards, e.file_offset,
                     &fd_idx, &base_off);
    const uint64_t file_off = base_off + (uint64_t)e.prp_idx * sub_io;

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
                         sub_io);
    } else {
        submit_write_one(dh,
                         e.prp_entry->prp1,
                         e.prp_entry->prp2,
                         file_off,
                         sub_io);
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
