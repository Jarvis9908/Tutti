#ifndef __TUTTI_NVME_STORAGE_NVME_FILE_DEVICE_HANDLE_H__
#define __TUTTI_NVME_STORAGE_NVME_FILE_DEVICE_HANDLE_H__

/**
 * nvme_file_device_handle.h -- on-GPU file handle.
 *
 * Layer: nvme_storage (R5b).
 *
 * Lives entirely in GPU memory.  Produced by
 * `INvmeStorage::acquire_device_handle(NvmeFile*)` (which cudaMallocs one
 * and cudaMemcpys a host-side template into it), consumed verbatim
 * by `__device__ submit_read_one / submit_write_one`.  Never
 * mutated after construction; freed by
 * `INvmeStorage::release_device_handle`.
 *
 * Extent storage -- inline-small + overflow (sizing note):
 *
 *   The per-file extent count is capped at `kNvmeFileHeaderMaxExtents`
 *   (124) at create time (fiemap_helper).  R11.5 removed the on-disk
 *   `NvmeFileHeader`; the cap is retained to bound log-entry size and
 *   kept under its historical name.  The GPU-resident handle is the
 *   opposite: every inline extent is real GPU memory that gets paid
 *   *every acquire*, and Tutti expects to hold potentially millions
 *   of these concurrently (LMCache-shaped KV-cache workloads).  So
 *   this struct decouples the two caps:
 *
 *     - `kNvmeFileDeviceHandleInlineExtents` (8) extents live inline.
 *       KV-cache files are pre-allocated, single-`fallocate()` fixed-
 *       size blocks; on an unfragmented (or lightly fragmented) NVMe
 *       namespace that yields 1-3 extents in practice, so 8 inline
 *       slots covers the overwhelming majority of real files with
 *       zero extra allocation.
 *     - Files that land beyond the inline cap (rare: heavy
 *       fragmentation) get a small `extents_overflow` GPU buffer,
 *       cudaMalloc'd to exactly `num_extents - inline_cap` entries.
 *       Only those pathological files pay for their own fragmentation;
 *       everyone else stays cheap.
 *
 *   This mirrors NVMe's own PRP1/PRP2 + PRP-list pattern (small
 *   inline, list only when needed) and is the fix for the "2M KV
 *   files at once" GPU-memory sizing problem: with 8 inline extents
 *   sizeof(NvmeFileDeviceHandle) is ~190-200 bytes (vs. the naive
 *   124-inline-extent design's ~2 KiB) -- an order of magnitude
 *   smaller, and comparable to or smaller than the legacy GeminiFS
 *   on-disk `geminiFS_hdr` (512 B cap, ~30 extents) that this
 *   supersedes.
 *
 *   Raising `kNvmeFileDeviceHandleInlineExtents` is a safe, purely
 *   size/cost tradeoff -- it does not change which files can be
 *   created or opened (the disk-side cap is independent and
 *   unaffected).  It only changes how many "typical" files fit
 *   without touching the overflow path.
 *
 * Why this struct is otherwise POD with inline fields:
 *
 *   1. Kernels read it in a single load for the common case -- no
 *      chasing pointers for the inline extents.
 *   2. cudaMemcpy is one shot for the base struct; only pathological
 *      files pay a second cudaMalloc+cudaMemcpy for the overflow.
 *   3. `extents_overflow` ownership is tracked host-side by
 *      HostFsBackedNvmeStorage (see host_fs_backed_nvme_storage.h);
 *      release_device_handle frees it there so this header stays a
 *      pure device-visible POD with no host-side bookkeeping baked in.
 *
 * What this struct deliberately does NOT contain:
 *
 *   - tensor / IOVA / PRP descriptors (memory/'s job; passed in
 *     per-call as prp1/prp2 args)
 *   - file_offset/length state (per-call args, not file-state)
 *   - back-pointer to the host-side NvmeFile (the GPU has no
 *     reason to dereference host pointers)
 */

#include <cstdint>

#include "lba_extent.h"
#include "nvme_file_header.h"   // kNvmeFileHeaderMaxExtents (disk-side cap)

// Forward declaration of libnvm's QueuePair.  We don't include
// queue.h here because queue.h drags in <cuda_runtime.h> and a chunk
// of host-only libnvm machinery; this header is meant to be
// includable from both .cpp and .cu callers.  The .cu submit
// helpers that actually dereference QueuePair (queue_acquire_helper.cuh,
// nvme_storage_device.cuh) include queue.h directly.
struct QueuePair;

namespace tutti {

/// GPU-resident inline extent cap.  Independent of (and much smaller
/// than) the disk-side `kNvmeFileHeaderMaxExtents` -- see the file
/// comment above for the sizing rationale.  Files whose FIEMAP extent
/// count exceeds this still work: the excess lands in a cudaMalloc'd
/// `extents_overflow` buffer built at acquire_device_handle() time.
inline constexpr uint32_t kNvmeFileDeviceHandleInlineExtents = 8;

struct NvmeFileDeviceHandle {
    /// PersistentFileLog::Entry::file_id of the source NvmeFile.
    /// Useful for debugging; kernels usually don't read this.
    uint64_t   file_id;

    /// Logical (user-visible) size in bytes.  R11.5: no in-band
    /// header, so this equals the on-disk file size.  IO requests
    /// with logical_off + nbytes > logical_size_bytes MUST be
    /// rejected by the caller; submit_*_one assumes the args are
    /// in-range.
    uint64_t   logical_size_bytes;

    /// Bytes to add to the user-supplied `logical_off` before
    /// walking extents, so users see a clean "offset 0 == first
    /// byte of payload".  R11.5: always 0 (no in-band header);
    /// the field is kept for kernel-source stability and
    /// resolve_lba still adds it (a no-op add of 0).
    uint32_t   header_bytes;

    /// NVMe block size for the namespace this file lives on
    /// (typ. 4096).  All `nbytes` and `logical_off` arguments
    /// MUST be multiples of this; submit_*_one converts both
    /// to LBA / block-count by right-shifting by
    /// `nvme_block_size_log`.
    uint32_t   nvme_block_size;
    uint32_t   nvme_block_size_log;   // log2(nvme_block_size)

    /// NVMe namespace id used in every SQE we synthesize.  Mirrors
    /// `qp->nvmNamespace` for the queue pairs in `d_qps` below.
    uint32_t   namespace_id;

    /// Total extent count (may exceed kNvmeFileDeviceHandleInlineExtents;
    /// never exceeds the disk-side kNvmeFileHeaderMaxExtents, enforced
    /// at acquire_device_handle time).
    uint32_t   num_extents;

    /// First min(num_extents, kNvmeFileDeviceHandleInlineExtents)
    /// extents, inline.  The common case (see file comment) never
    /// touches `extents_overflow`.
    LbaExtent  extents[kNvmeFileDeviceHandleInlineExtents];

    /// GPU-resident pointer to the remaining
    /// `num_extents - kNvmeFileDeviceHandleInlineExtents` extents,
    /// contiguous, in FIEMAP order continuing where `extents[]` left
    /// off.  nullptr when num_extents <= kNvmeFileDeviceHandleInlineExtents
    /// (the overwhelmingly common case).  Owned + freed by
    /// HostFsBackedNvmeStorage::release_device_handle; kernels only
    /// ever read through it.
    LbaExtent* extents_overflow;

    /// Back-pointer to the controller's d_qps[] pool.  Lives on
    /// the GPU memory of the same cuda device as this handle.
    /// Lifetime is tied to the LocalNvmeDevice's queue_group: as long
    /// as that Controller is alive, d_qps is valid.  Kernels MUST
    /// NOT release / cudaFree this pointer.
    QueuePair* d_qps;
    uint32_t   num_d_qps;

    /// Reserved for future use.  No exact-size ABI guarantee is
    /// needed here (unlike the on-disk NvmeFileHeader): this struct
    /// is built + cudaMemcpy'd fresh by the same process that reads
    /// it, never persisted or shared across a version boundary.
    uint32_t   reserved0;
};

} // namespace tutti

#endif // __TUTTI_NVME_STORAGE_NVME_FILE_DEVICE_HANDLE_H__
