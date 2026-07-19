#ifndef __TUTTI_IO_ENGINE_LOCAL_NVME_NVME_BATCH_H__
#define __TUTTI_IO_ENGINE_LOCAL_NVME_NVME_BATCH_H__

/**
 * nvme_batch.h -- NVMe-native batch IO entry types.
 *
 * Layer: io_engine, local_nvme backend.
 *
 * NvmeBatchEntry carries the FULL shard pointer table per IO --
 * NOT a pre-resolved single shard.  The kernel does its own stripe
 * selection (`fd_idx = gpu_blk % num_shards`) the same way legacy
 * `nvme_batch_xfer_kernel` did, then fans out into R5b's
 * `submit_{read,write}_one`.  Hosting the shard choice on the GPU
 * is what lets one tensor's sub-slices land on multiple shards in
 * the same batch without exploding the host-side entry count.
 *
 * Memory residency:
 *   - NvmeBatchEntry is residency-neutral (POD).  Builders fill an
 *     array on the host; the array is then cudaMemcpy'd to GPU
 *     before kernel launch.  All pointers it carries
 *     (`shards`, `prp_entry`) point to GPU-resident objects --
 *     the host code stores them as opaque values without dereferencing.
 *
 * Lifetime contract:
 *   - The pointed-to shard table (`GpuFileHandle::d_shards_dev`)
 *     and the pointed-to AddressDescriptor (R7
 *     `IoSliceView::d_ios`) MUST stay valid until the matching
 *     kernel completes.  Caller manages this via R6
 *     release_device_handle / R7 unregister.
 */

#include <cstdint>

namespace tutti {

struct AddressDescriptor;       // memory/include/memory_subsystem.h
struct NvmeFileDeviceHandle;    // nvme_storage/include/nvme_file_device_handle.h
struct GpuFileHandle;           // block_storage/include/block_storage.h
struct MemoryRegion;            // memory/include/memory_region.h
class  IMemorySubsystem;        // memory/include/memory_subsystem.h

/// One IO worth of context.  Map to legacy GPUIoContext line-by-line:
///
///   legacy field          | NvmeBatchEntry field
///   ----------------------+-----------------------
///   nvme_files (span)     | shards + num_shards
///   prp_entry             | prp_entry
///   prp_idx               | prp_idx
///   file_offset           | file_offset
///
/// `is_read` is per-entry today purely to keep mixed-direction
/// batches feasible if we ever need them; the v0.1 R8 kernel reads
/// the batch-uniform `is_read` argument and ignores the per-entry
/// field.  Builders fill it consistently with the batch direction
/// so a future per-entry kernel will Just Work.
///
/// Sizing: 8 + 4 + 8 + 4 + 8 + 1 + 7(pad) = 40 bytes, naturally aligned.
struct NvmeBatchEntry {
    NvmeFileDeviceHandle* const* shards;        ///< GPU-resident shard ptr table,
                                                ///<   length == num_shards.
                                                ///<   Equals the GpuFileHandle's
                                                ///<   d_shards_dev for the input
                                                ///<   tensor's GpuFile.
    uint32_t                     num_shards;    ///< == GpuFileHandle::num_shards.

    const AddressDescriptor*     prp_entry;     ///< Points into the IoSliceView's
                                                ///<   GPU-resident d_ios[] for this
                                                ///<   sub-slice.
    uint32_t                     prp_idx;       ///< Sub-slice index *within the
                                                ///<   tensor* (legacy semantics:
                                                ///<   global, NOT slice-local).
                                                ///<   Used by the kernel to compute
                                                ///<   gpu_blk = file_offset/blk_size
                                                ///<            + prp_idx.

    uint64_t                     file_offset;   ///< The tensor's BASE byte offset
                                                ///<   within the GpuFile.  Same
                                                ///<   value across every entry
                                                ///<   that targets this tensor;
                                                ///<   the kernel adds prp_idx-
                                                ///<   driven offsets on top.

    bool                         is_read;       ///< Reserved for per-entry direction
                                                ///<   (R8 kernel today reads the
                                                ///<   batch-uniform parameter).
    uint8_t                      pad[7];
};

/// Caller-owned batch view.  Today the batch is uniform-direction
/// (`is_read`); per-entry direction is reserved by the entry struct
/// for future use.
///
/// `d_entries` MUST be GPU-resident before `nvme_batch_xfer_kernel`
/// is launched.  The host_batch_builder fills a host-side array; the
/// caller cudaMemcpy's it to a device region.
struct NvmeBatchRequest {
    const NvmeBatchEntry* d_entries;
    uint32_t              count;
    bool                  is_read;
};

/// One input row to the host_batch_builder.  Caller bundles a
/// memory-registered tensor with the GpuFile shard handle table and
/// the tensor's base byte offset within that GpuFile.
struct NvmeBatchInputTensor {
    MemoryRegion*  tensor_region;       ///< R7 register_tensor result;
                                        ///<   provides the IoSliceView[].
    GpuFileHandle* file_handle;         ///< R6 acquire_device_handle result;
                                        ///<   provides d_shards_dev + num_shards.
    uint64_t       file_byte_offset;    ///< Tensor base in GpuFile.  Legacy
                                        ///<   batched-xfer formula:
                                        ///<     layer_idx * tensor_size * 2
                                        ///<     [+ tensor_size for V-cache].
};

} // namespace tutti

#endif // __TUTTI_IO_ENGINE_LOCAL_NVME_NVME_BATCH_H__
