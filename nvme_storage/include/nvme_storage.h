#ifndef __TUTTI_NVME_STORAGE_NVME_STORAGE_H__
#define __TUTTI_NVME_STORAGE_NVME_STORAGE_H__

/**
 * nvme_storage.h -- the INvmeStorage interface.
 *
 * Layer: nvme_storage.  See doc/refactor/LegacyDecomposition.md §3.6.
 *
 * Role:
 *   - Owns named LBA ranges on top of one or more NVMe namespaces.
 *   - Knows how to mount / format / FIEMAP host filesystems on top
 *     of the snvme block device, but exposes only a flat "named
 *     file" API.  No filesystem-specific concept leaks out.
 *   - Provides host-side blocking IO (for bootstrap / metadata /
 *     tests) and -- in a follow-up R5b -- device-side submission
 *     primitives that io_engine kernels call inline.
 *
 * Layer boundary:
 *   - This header MUST NOT pull in libnvm headers; we want callers
 *     (block_storage) to consume the abstraction without leaking
 *     libnvm types.  Implementations are free to use libnvm
 *     internally.
 *   - Capacity / file directory metadata is persisted on the host
 *     filesystem (via PersistentFileLog).  That backing store is
 *     deliberately not exposed.
 *
 * Lifetime:
 *   - bootstrap() takes a list of Device*; for each device it
 *     mounts the host fs, prepares a directory, and reloads the
 *     PersistentFileLog.  Failure rolls back partial mounts.
 *   - shutdown() flushes everything and unmounts.  Idempotent.
 *
 * Multi-NVMe:
 *   - One INvmeStorage instance services multiple devices.
 *     Per-Device state is keyed off `Device*` everywhere.
 *
 * Threading:
 *   - Public methods are thread-safe; v0.1 uses a single std::mutex.
 *
 * Deferred to R5b: (DONE)
 *   - acquire_queue_pair / release_queue_pair: subsumed by giving
 *     out NvmeFileDeviceHandle that already references the
 *     controller's d_qps[] pool.  Host never touches a queue pair
 *     by handle.
 *   - __device__ submit_read_one / submit_write_one: in
 *     `nvme_storage_device.cuh`, callable from any kernel.
 *   - NvmeFileDeviceHandle: in `nvme_file_device_handle.h`,
 *     produced by acquire_device_handle() below.
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <sys/types.h>     // ssize_t
#include <vector>

#include "nvme_file.h"

namespace tutti {

struct Device;
struct NvmeFileDeviceHandle;       // nvme_file_device_handle.h

/// Flags for the unified open_file() below -- POSIX open(2)-shaped:
/// EXISTING (0, the default) behaves like plain open(); CREATE behaves
/// like O_CREAT (create if missing, open if it already exists);
/// CREATE|EXCL behaves like O_CREAT|O_EXCL (fail if it already
/// exists).  NO_PERSIST/NO_SYNC are Tutti-specific bulk-init knobs
/// (see open_file's doc comment) with no POSIX analogue.
enum NvmeOpenFlags : uint32_t {
    NVME_OPEN_EXISTING   = 0,        ///< fail if the file doesn't exist
    NVME_OPEN_CREATE     = 1u << 0,  ///< create if missing (~O_CREAT)
    NVME_OPEN_EXCL       = 1u << 1,  ///< with CREATE: fail if it already exists (~O_EXCL)
    NVME_OPEN_NO_PERSIST = 1u << 2,  ///< with CREATE: defer log persist (bulk init)
    NVME_OPEN_NO_SYNC    = 1u << 3,  ///< with CREATE: defer fsync (bulk init)
};

/// Opaque stream handle -- lets this CUDA-free header talk about
/// "queue async GPU work on this stream" without pulling in
/// <cuda_runtime.h> (see the layer-boundary rule in the file
/// comment above).  Callers that already hold a `cudaStream_t`
/// (Coordinator, adapters -- anything CUDA-aware) pass it through as
/// `(GpuStreamHandle)my_stream`; the .cu implementation
/// `reinterpret_cast`s it back.
using GpuStreamHandle = void*;

class INvmeStorage {
public:
    virtual ~INvmeStorage() = default;

    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    /// Mount + prepare the on-disk directory for every device.
    /// Returns false on any failure; partial state rolled back.
    /// MUST be called before any other method.
    virtual bool bootstrap(const std::vector<const Device*>& devices) = 0;

    /// Unmount everything and release internal state.  Safe to call
    /// twice.  Returns false if any unmount failed (state is still
    /// cleared on best-effort).
    virtual bool shutdown() = 0;

    // ------------------------------------------------------------------
    // Capacity (per device)
    // ------------------------------------------------------------------

    /// Total size in bytes of the host filesystem backing `device`.
    /// Returns 0 if `device` is not bootstrapped.
    virtual uint64_t total_capacity   (const Device*) const = 0;

    /// Free space in bytes on the host filesystem backing `device`.
    /// Returns 0 if `device` is not bootstrapped.
    virtual uint64_t available_capacity(const Device*) const = 0;

    /// Where this Device's host filesystem is mounted (so callers
    /// can place sidecar metadata under <mount>/.tutti/).  Returns
    /// empty string if `device` is not bootstrapped.  Used by
    /// block_storage to colocate gpu_file_log.bin with
    /// nvme_storage's file_log.bin under the same .tutti directory.
    virtual std::string mount_path(const Device*) const = 0;

    // ------------------------------------------------------------------
    // Directory
    // ------------------------------------------------------------------

    /// POSIX open(2)-shaped: open an existing NvmeFile, or (with
    /// `NVME_OPEN_CREATE`) create it if it doesn't exist yet.
    ///
    ///   flags == NVME_OPEN_EXISTING (default): open an existing file.
    ///            Returns nullptr if not found.
    ///
    ///   flags & NVME_OPEN_CREATE: if the file doesn't exist, create
    ///            it -- allocates `create_size` bytes on the host
    ///            filesystem (no in-band header prefix; user
    ///            data starts at byte 0), stores the FIEMAP extent
    ///            list in the persistent file log, and creates a
    ///            .refs/ hardlink as an inode refcount -- then open
    ///            it exactly like the EXISTING path would.  If it
    ///            ALREADY exists, behaves like EXISTING (`create_size`
    ///            is ignored) UNLESS `NVME_OPEN_EXCL` is also set, in
    ///            which case that's a failure (nullptr), mirroring
    ///            O_CREAT|O_EXCL.
    ///
    ///   Returns nullptr on: not-found (EXISTING), already-exists
    ///   (CREATE|EXCL), or any allocation/FIEMAP/host-fs failure.
    ///
    /// Bulk-init knobs (only meaningful with CREATE; default when
    /// unset = "single-file durable"):
    ///   NVME_OPEN_NO_SYNC     skip the fsync a create normally does
    ///                         (after fallocate; there is only one
    ///                         fsync, since there is no in-band header
    ///                         pwrite).  The file's extents
    ///                         stay in the page cache but are NOT yet
    ///                         on platter.  Caller MUST follow up with
    ///                         `flush_metadata()` (syncfs()s the mount)
    ///                         before relying on durability.  Crash
    ///                         without that flush leaves a ghost .bin
    ///                         which bootstrap reconcile unlinks on the
    ///                         next start, so re-init must be
    ///                         idempotent.
    ///   NVME_OPEN_NO_PERSIST  keep the new directory entry in memory
    ///                         only; a later `flush_metadata()`
    ///                         rewrites the log once for all
    ///                         accumulated changes instead of once per
    ///                         call (collapses O(N^2) total bytes
    ///                         written across N creates into O(N)).
    ///
    /// Bulk-init pattern for, e.g., LMCache provisioning of millions
    /// of pre-allocated KV-shard files.  close_file is a no-op
    /// (NvmeFile is metadata-only, stays resident until delete_file),
    /// so the close_file call below is optional -- included for
    /// symmetry with the non-bulk single-file pattern:
    /// @code
    /// for (auto& s : shards) {
    ///     NvmeFile* nf = storage->open_file(dev, s.name, s.size,
    ///         NVME_OPEN_CREATE | NVME_OPEN_NO_PERSIST | NVME_OPEN_NO_SYNC);
    ///     storage->close_file(nf);
    /// }
    /// storage->flush_metadata(dev);   // one syncfs + one log rewrite
    /// @endcode
    virtual NvmeFile* open_file(const Device*    device,
                                std::string_view name,
                                uint32_t         flags       = NVME_OPEN_EXISTING,
                                uint64_t         create_size = 0) = 0;

    /// One entry of a batch create (see open_files_batch).
    struct CreateSpec {
        const Device* device;
        std::string   name;
        uint64_t      size_bytes;
    };

    /// Batch create `count` files CONCURRENTLY.  Spawns worker
    /// threads internally; each file's FS operations (open/fallocate/
    /// fiemap/linkat) run WITHOUT the storage mutex (only the
    /// in-memory bookkeeping is serialized), so the threads genuinely
    /// parallelize at the NVMe level.  `flags` is applied to every file
    /// (typically NVME_OPEN_CREATE | NVME_OPEN_NO_PERSIST |
    /// NVME_OPEN_NO_SYNC for bulk init).  out[i] corresponds to
    /// specs[i]; nullptr on that file's failure.  Caller MUST call
    /// flush_metadata(device) once per device after the batch.
    /// Returns false if any file failed (out[] still filled for the
    /// successes).
    virtual bool open_files_batch(const CreateSpec* specs, uint32_t count,
                                  uint32_t flags, NvmeFile** out) = 0;

    /// Flush all deferred metadata for `device`:
    ///   - syncfs() the mount so any data/extents written by
    ///     `open_file(..., NVME_OPEN_NO_SYNC)` since the last flush
    ///     reach the platter.
    ///   - rewrite + fsync + rename the PersistentFileLog so any
    ///     entries added by `open_file(..., NVME_OPEN_NO_PERSIST)`
    ///     since the last flush become durable.
    /// No-op if nothing is pending; safe to call repeatedly.
    /// Returns false if either step fails (state still considered
    /// dirty, retry is OK).
    virtual bool      flush_metadata(const Device* device) = 0;

    /// No-op.  NvmeFile is metadata-only (no held fd) and stays
    /// resident until delete_file; there is nothing to close or flush.
    /// Durability of written data is the caller's responsibility via
    /// flush_metadata(device) / sync(file).
    virtual bool      close_file(NvmeFile* file) = 0;

    /// Remove the file from the directory + delete the underlying
    /// host file.  Returns false if not found or unlink fails.
    ///
    /// `persist_now` (default true) controls whether the per-Device
    /// log is rewritten before this call returns.  Set false for
    /// bulk deletion (e.g. fsck-style cleanup of millions of stale
    /// files), then call `flush_metadata(device)` once at the end
    /// to land all log changes in a single rewrite.  The on-disk
    /// `<name>.bin` file IS unlinked synchronously regardless of
    /// this flag -- only the log update is deferred.
    virtual bool      delete_file(NvmeFile* file,
                                  bool      persist_now = true) = 0;

    /// Batch delete: unlinks `count` files CONCURRENTLY (worker
    /// threads, same split-lock shape as open_files_batch -- the two
    /// unlink(2) calls + cache-erase per file run WITHOUT the storage
    /// mutex; only the in-memory bookkeeping, i.e. `files` map erase
    /// + log->remove, is serialized).  Always defers the log rewrite
    /// (as if persist_now=false); caller MUST call
    /// flush_metadata(device) once per device after the batch.
    /// out_ok[i] corresponds to files[i] (true on success); returns
    /// false if any file failed (out_ok still filled for every i).
    virtual bool      delete_files_batch(NvmeFile* const* files,
                                         uint32_t         count,
                                         bool*            out_ok) = 0;

    /// All currently-resident NvmeFiles for `device` (i.e. entries
    /// that have been opened/created in this process).  This is NOT
    /// the same as "every file in the persistent directory": entries
    /// on disk that have never been opened in this process do NOT
    /// appear.  Use `list_file_names()` for the directory-wide view.
    virtual std::vector<NvmeFile*> list_files(const Device*) const = 0;

    /// All file names known to this device's PersistentFileLog,
    /// regardless of open-status in this process.  Useful for bulk
    /// cleanup / fsck / migration: caller can `open_file(name)` each
    /// then `delete_file()` or whatever.  Order is the log's
    /// insertion order.  No file is opened into memory by this call.
    virtual std::vector<std::string> list_file_names(const Device*) const = 0;

    // ------------------------------------------------------------------
    // Host-side blocking IO
    //
    // Useful for:
    //   - bootstrap / metadata writes that don't need GPU paths
    //   - the smoke test (verify byte content roundtrip)
    //   - cooperative CPU side of mixed CPU+GPU IO
    //
    // Implementation opens a temporary O_DIRECT fd via the .refs/
    // hardlink (pread/pwrite) per call and closes it immediately --
    // data bypasses the page cache entirely.  It is NOT the high-
    // throughput path -- that's the GPU device-side submit below.
    //
    // `byte_offset` is the LOGICAL offset (relative to byte 0 of user
    // data; no header prefix, data_offset is 0).  O_DIRECT
    // requires byte_offset, len, and the buffer address to all be
    // block-aligned (EINVAL otherwise).
    // ------------------------------------------------------------------

    virtual ssize_t read_blocking (NvmeFile* file, uint64_t byte_offset,
                                    void*        dst, size_t len) = 0;
    virtual ssize_t write_blocking(NvmeFile* file, uint64_t byte_offset,
                                    const void* src, size_t len) = 0;

    /// fsync the file (data + metadata).  Useful when callers want
    /// a flush point without closing.
    virtual bool    sync(NvmeFile* file) = 0;

    // ------------------------------------------------------------------
    // GPU device-side submit
    //
    // These DO NOT open or create a file -- the file is already open
    // (via `open_file` above).  Acquire / release a GPU-side *view*
    // (a small POD living in GPU memory) that lets a kernel submit
    // reads/writes against the file's LBA extents through the NVMe
    // queue group's d_qps.  Naming intentionally avoids "open" so
    // this isn't confused with `open_file` itself.
    //
    // Handles live in a two-tier cache per cuda device (see
    // memory/include/tiered_handle_cache.h): a large CPU-pinned L2
    // (every touched file's built template, FIEMAP work done once)
    // backs a small GPU-resident L1 (the current working set).  A
    // single GPU-resident pool sized to hold every file's handle
    // would cost GBs of scarce GPU memory at LMCache scale (10s of
    // millions of files); the two-tier split keeps GPU residency
    // bounded to the working set while amortizing FIEMAP extent work
    // across the whole file population.  L1 eviction is a downgrade
    // (content survives in L2, next promotion is a plain memcpy,
    // never re-walks FIEMAP); only L2 eviction (rare, given its large
    // budget) is a genuine delete.
    //
    //   Consequence for this API: acquire_device_handle is
    //   idempotent / safe to call repeatedly for the same file (cache
    //   hit is nearly free) -- it is "ensure resident", not "allocate
    //   once".  release_device_handle takes the NvmeFile* itself, not
    //   the handle pointer (GPU addresses are recycled across tiers
    //   and no longer uniquely identify a file), and means "you may
    //   downgrade this from L1 now", a hint the LRU would eventually
    //   act on anyway.
    //
    // configure_handle_pool  MUST be called once, after bootstrap()
    //                        and before the first acquire_device_handle,
    //                        to size the per-cuda-device
    //                        TieredHandleCache<NvmeFileDeviceHandle,
    //                        NvmeFileId> that backs every subsequent
    //                        acquire.  `l1_capacity` is the max number
    //                        of DISTINCT files whose handles may be
    //                        concurrently GPU-resident; `l2_capacity`
    //                        is the max number whose built template
    //                        may be CPU-resident (>= l1_capacity).
    //                        See coordinator_config.h's
    //                        handle_l1_gpu_budget_bytes /
    //                        handle_l2_host_budget_bytes for how the
    //                        caller typically derives these numbers.
    //                        Calling it twice, or after any acquire,
    //                        is a logic error.
    //
    // acquire_device_handle   Ensures `file`'s handle is GPU-resident
    //                         (cache hit: near-free LRU touch; L2 hit:
    //                         one memcpy promotion; COLD miss: builds
    //                         the template -- walks FIEMAP -- then
    //                         promotes).  Returns the device pointer
    //                         immediately; content is guaranteed
    //                         valid for GPU work queued on `stream`
    //                         immediately after this call (the
    //                         implementation internally fences any
    //                         cross-stream reuse -- callers never need
    //                         to reason about which stream last filled
    //                         a recycled slot).
    //
    //                         Returns nullptr if:
    //                           - the device has no NvmeQueueGroup
    //                             (its registry was opened with
    //                             build_queue_group=false).
    //                           - configure_handle_pool was never
    //                             called.
    //                           - the FIEMAP/extent build failed.
    //
    // release_device_handle   Hints that `file` may be downgraded
    //                         from L1 to L2 now (content preserved,
    //                         GPU slot freed for reuse).  Purely
    //                         advisory -- the LRU would eventually
    //                         evict it anyway; calling this just lets
    //                         it happen sooner.  No-op if `file` isn't
    //                         currently L1-resident.
    //
    // Lifetime: the handle is valid as long as the underlying
    // NvmeFile is alive AND the device's queue_group is alive.  In
    // practice that means: don't outlive the storage subsystem
    // shutdown.
    //
    // R6 note: the unified "POSIX-style open that yields both host
    // and GPU views" lives at the block_storage layer; this acquire
    // pair is its building block.
    // ------------------------------------------------------------------

    virtual bool configure_handle_pool(uint32_t l1_capacity, uint32_t l2_capacity) = 0;

    virtual NvmeFileDeviceHandle* acquire_device_handle (NvmeFile*       file,
                                                         GpuStreamHandle stream) = 0;
    virtual void                  release_device_handle(NvmeFile*       file,
                                                         GpuStreamHandle stream) = 0;

    /// Batch variant of acquire_device_handle: ensures every file in
    /// `files[0..count)` is GPU-resident with (where possible) a
    /// single contiguous H2D copy for the COLD subset, instead of
    /// `count` separate small copies -- see TieredHandleCache's batch
    /// design.  `out_handles[i]` corresponds to `files[i]`.
    ///
    /// Hard requirement (enforced by TieredHandleCache): `count` MUST
    /// be <= this call's l1_capacity, because every handle resolved
    /// by one batch call must stay L1-resident for the duration of
    /// the call (mirrors max_entries_per_batch's contract).  Returns
    /// false (whole batch failed) on any per-file failure.
    virtual bool acquire_device_handles_batch(NvmeFile* const*      files,
                                              uint32_t              count,
                                              GpuStreamHandle       stream,
                                              NvmeFileDeviceHandle** out_handles) = 0;

    /// Pre-load `file`'s handle template into the L2 (CPU-pinned)
    /// metadata cache WITHOUT promoting to L1 (GPU-resident).  Useful
    /// for bulk pre-warming after a bulk-create: call this for each
    /// file so the first acquire_device_handle is a fast L2->L1
    /// promotion (one memcpy) instead of a cold build (FIEMAP walk +
    /// overflow cudaMalloc).  No-op if the handle pool was never
    /// configured (pure host-side users) or the file's Device has no
    /// queue_group.  Admission is lazy -- open_file does not call this
    /// automatically; callers who want pre-warming call it explicitly.
    virtual void admit_to_cache(NvmeFile* file) = 0;

    /// Cumulative two-tier handle-cache counters, summed across every
    /// per-cuda-device cache.  Lets stress tests prove eviction
    /// actually happened and measure its cost.  All-zero if the handle
    /// pool was never configured.
    struct CacheStats {
        uint64_t cold_builds   = 0;  // COLD miss -> handle template built
        uint64_t l1_hits       = 0;  // already GPU-resident
        uint64_t l2_hits       = 0;  // L2-resident, promoted to L1
        uint64_t l1_promotions = 0;  // L2 -> L1 (memcpy into GPU)
        uint64_t l1_evictions  = 0;  // L1 -> L2 downgrade (GPU slot freed)
        uint64_t l2_evictions  = 0;  // L2 genuine delete (rebuild next time)
    };
    virtual CacheStats cache_stats() const = 0;
};

} // namespace tutti

#endif // __TUTTI_NVME_STORAGE_NVME_STORAGE_H__
