#ifndef __TUTTI_BLOCK_STORAGE_BLOCK_STORAGE_H__
#define __TUTTI_BLOCK_STORAGE_BLOCK_STORAGE_H__

/**
 * block_storage.h -- the IBlockStorage interface (R6).
 *
 * Layer: block_storage.  Sits *above* nvme_storage and *below*
 * io_engine.  See doc/refactor/LegacyDecomposition.md §3.4.
 *
 * Role:
 *   - Owns the "GpuFile" abstraction: a named logical container
 *     whose data plane is split across N NvmeFile shards (one per
 *     entry in `shard_placement`, currently 1:1 with Devices).
 *   - Persists GpuFile metadata (id, name, tensor_shape, shard
 *     names) per device in `<mount>/.tutti/gpu_file_log.bin`.
 *     Every device that hosts at least one shard carries a full
 *     copy of that log so a single-disk loss does not erase the
 *     directory view.
 *   - Provides bulk-init paths that delegate to nvme_storage's
 *     bulk-init flags (create_file persist_now=false, sync_now=false
 *     followed by flush_metadata) so creating millions of GpuFiles
 *     at LMCache provisioning time stays sub-minute.
 *   - Provides `acquire_device_handle` / `release_device_handle` to
 *     bring all N shard NvmeFiles to GPU together; the returned
 *     GpuFileHandle is what io_engine (R8) consumes when it
 *     pre-stages a kernel batch (one GPUIoContext per IO; each ctx
 *     references handle->d_shards_dev so the kernel reads it
 *     directly without chasing host pointers).
 *
 * Layer boundary:
 *   - This header MUST NOT pull in libnvm or CUDA headers; we want
 *     io_engine and smoke code to see a small, libnvm-free surface.
 *     (NvmeFileDeviceHandle is forward-declared; its full layout
 *     lives in nvme_storage/include/nvme_file_device_handle.h.)
 *   - The IO-submission hot path is NOT here.  This header only
 *     gives you metadata + acquired handles; the kernel-side
 *     translation/submit step lives in io_engine (R8) on top of
 *     gpu_file_resolve.h + nvme_storage_device.cuh.
 *
 * Lifetime:
 *   - bootstrap() takes a non-owning `INvmeStorage*` and the device
 *     list to participate in.  It loads each device's
 *     gpu_file_log.bin (a generation-number-arbitrated mirror) and
 *     reconciles it against nvme_storage's .bin directory.
 *   - shutdown() closes all open GpuFiles and persists the log on
 *     every device.  Idempotent.
 *
 * Multi-NVMe placement (v0.1):
 *   - GpuFileSpec.shard_placement explicitly enumerates which
 *     Device each shard lives on.  size() must equal
 *     tensor_shape[0].  v0.1 is 1:1 -- one shard per device, no
 *     two shards on the same device per file.  Higher-level
 *     placement policies (e.g. "balance shards across devices to
 *     fill the lowest-utilised pair first") are caller-side
 *     concerns above this layer.
 *
 * Threading:
 *   - Public methods are thread-safe; v0.1 uses a single std::mutex.
 *
 * The "block" name is historical (LegacyDecomposition.md §3.4).  In
 * the real world it always represents a GPU file; the surface
 * deliberately uses GpuFile / GpuFileHandle rather than Block /
 * BlockHandle.
 */

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace tutti {

struct Device;                  // device_manager
struct NvmeFile;                // nvme_storage/include/nvme_file.h
struct NvmeFileDeviceHandle;    // nvme_storage/include/nvme_file_device_handle.h
class  INvmeStorage;            // nvme_storage/include/nvme_storage.h

using GpuFileId = uint32_t;

/// Flags for the unified open_gpu_file() below -- mirrors
/// nvme_storage.h's NvmeOpenFlags (POSIX open(2)-shaped).  EXISTING
/// (0, the default) opens an already-created GpuFile by
/// `spec.name` alone (every other GpuFileSpec field is ignored);
/// CREATE additionally allocates one if `spec.name` doesn't exist
/// yet (needs the full spec).  NO_PERSIST has no POSIX analogue --
/// it's Tutti's bulk-init knob.
enum GpuFileOpenFlags : uint32_t {
    GPU_FILE_OPEN_EXISTING   = 0,        ///< fail if the GpuFile doesn't exist
    GPU_FILE_OPEN_CREATE     = 1u << 0,  ///< create if missing (~O_CREAT)
    GPU_FILE_OPEN_EXCL       = 1u << 1,  ///< with CREATE: fail if it already exists (~O_EXCL)
    GPU_FILE_OPEN_NO_PERSIST = 1u << 2,  ///< with CREATE: defer gpu_file_log persist (bulk init)
};

/// Opaque stream handle -- same idea as nvme_storage.h's
/// `GpuStreamHandle` (a distinct but layout-identical `void*` alias,
/// so this header keeps its own "no CUDA" promise without including
/// nvme_storage.h just for one typedef).  Callers holding a
/// `cudaStream_t` pass `(GpuStreamHandle)stream`; the .cpp
/// implementation `reinterpret_cast`s it back.
using GpuStreamHandle = void*;

/// Hard upper bound on the number of shards (== tensor_shape[0])
/// per GpuFile.  Set to 4 because GPUs are typically PCIe x16 and
/// modern NVMe are PCIe x4, so four parallel queues already saturate
/// the GPU's host link.  Higher values may need scattergather changes
/// in io_engine.  Bumping this requires re-running every smoke and
/// extending OnDiskGpuFileEntry::per_shard_names accordingly (it is
/// length-prefixed in the on-disk format so this is mechanical, not
/// an ABI break -- but it IS a re-test-everything event).
constexpr uint32_t kGpuFileMaxShards = 4;

/// Description used to create a new GpuFile.
struct GpuFileSpec {
    /// Persistent directory name; unique within the storage.
    std::string_view name;

    /// Total user-visible bytes.  Must equal
    ///   tensor_shape[0] * tensor_shape[1] * tensor_shape[2].
    /// Stored as the authoritative size; tensor_shape is metadata
    /// for callers (LMCache, etc.) and the dimensional check.
    uint64_t total_size;

    /// Three-tuple describing the GpuFile content layout.  Tutti's
    /// hot path only reads tensor_shape[0] (== num_shards) and
    /// tensor_shape[2] (== tensor_size, the per-tensor interleave
    /// unit).  The middle dimension tensor_shape[1] is metadata
    /// only -- typical values are "layers per shard" for KV-cache,
    /// but Tutti does not interpret it.
    ///
    /// tensor_shape[0] = num_shards         (in [1, kGpuFileMaxShards])
    /// tensor_shape[1] = layers_per_shard   (>= 1, opaque to runtime)
    /// tensor_shape[2] = tensor_size_bytes  (must divide total_size)
    ///
    /// Note: tensor_size is the LOGICAL stripe / interleave unit at
    /// the GpuFile layer; it is NOT the NVMe-level IO size.  When
    /// tensor_size exceeds the device's MDTS (Max Data Transfer
    /// Size, typ. 128 KiB), `memory/` (R7) will fragment each
    /// tensor at register_tensor() time into ceil(tensor_size /
    /// MDTS) PRPMappingEntry sub-slices, and `io_engine/` (R8)
    /// will issue that many GPUIoContext entries per tensor at
    /// submit time.  This layer does not need to know about that
    /// fan-out: gpu_file_resolve() returns the shard's BASE byte
    /// offset for the tensor, and R8 adds each sub-slice's
    /// intra-tensor offset on top.
    uint32_t tensor_shape[3];

    /// Where each shard's backing NvmeFile lives.  Element i is the
    /// device for shard i.  size() must equal tensor_shape[0].
    /// v0.1: pointers must be distinct (1:1 shard-to-device).
    std::vector<const Device*> shard_placement;
};

/// Host-side runtime view of an open GpuFile.
///
/// Lifetime:
///   - Created by IBlockStorage::open_gpu_file.
///   - Borrowed only; callers MUST NOT delete -- the storage owns
///     it and reclaims on close_gpu_file / shutdown.
///   - shards[] are NvmeFile* borrowed from nvme_storage (same
///     reference-borrow rules: don't outlive INvmeStorage::shutdown).
struct GpuFile {
    GpuFileId   id;
    std::string name;

    uint64_t    total_size;
    uint32_t    tensor_shape[3];

    /// Borrowed NvmeFile* per shard, parallel to
    /// GpuFileSpec::shard_placement.  size() == tensor_shape[0].
    std::vector<NvmeFile*> shards;
};

/// "Acquired" GpuFile: every shard has been brought to GPU.
///
/// Allocated by IBlockStorage::acquire_device_handle.
/// Released by IBlockStorage::release_device_handle (which also
/// cudaFrees `d_shards_dev` and calls release_device_handle on each
/// underlying NvmeFile).
///
/// The struct itself lives in *host* memory.  io_engine consumes
/// it during host-side per-IO context staging (see
/// LegacyDecomposition.md §R8): for each IO, host code does
///   gpu_file_resolve(handle->tensor_size, handle->num_shards,
///                    byte_off, &si, &so);
///   ctx.d_shard         = handle->d_shards_host[si];
///   ctx.shard_byte_off  = so;
///   ctx.length          = handle->tensor_size;
/// then cudaMemcpyAsync the ctx[] array onto GPU and launches one
/// thread per IO.  The kernel never touches `d_shards_dev`
/// directly; it only dereferences a single
/// `NvmeFileDeviceHandle*` per ctx.  We still expose
/// `d_shards_dev` as a convenience for any future kernel that
/// wants to do its own resolve on-device with a single ctx
/// describing the whole file.
struct GpuFileHandle {
    /// Borrowed back-pointer.  Keeps name / total_size / shards
    /// reachable for diagnostics; do not free.
    GpuFile* file;

    /// Cached for fast access (same values as
    /// `file->tensor_shape[2]` and `file->tensor_shape[0]`).
    ///
    /// tensor_size here is the logical stripe unit, NOT MDTS.
    /// io_engine (R8) computes per-IO sub-slice sizes from R7's
    /// PRPMappingEntry tables, not from this field; this field's
    /// only role on the hot path is the `gpu_file_resolve`
    /// computation that picks a shard.
    uint32_t tensor_size;
    uint32_t num_shards;

    /// Host-side table of GPU-resident shard handles, in the same
    /// order as `file->shards`.  Element i was produced by
    /// INvmeStorage::acquire_device_handle(file->shards[i]) and
    /// MUST be released via the matching release at handle teardown.
    ///
    /// Each pointer itself lives in GPU memory (cudaMalloc'd by
    /// nvme_storage); do NOT dereference on host.  The vector that
    /// holds them is host-resident.
    ///
    /// Hot-path usage (mirrors the legacy gpu_controller.cu design):
    /// io_engine (R8) host-stages a per-IO context array; for each
    /// IO it resolves shard_idx via gpu_file_resolve() and copies
    /// d_shards_host[shard_idx] *into the context* before
    /// cudaMemcpy'ing the array to GPU.  The kernel therefore only
    /// dereferences a single NvmeFileDeviceHandle* per IO -- no
    /// indirection through a GPU-resident shard table.  This is both
    /// simpler and avoids an extra GPU load per IO compared with a
    /// `d_shards_dev[shard_idx]` lookup.
    std::vector<NvmeFileDeviceHandle*> d_shards_host;

    /// GPU-resident shard pointer table (R8; async pooled R11).
    ///
    /// `d_shards_dev` points at one slot of a
    /// `GpuSlotPool<ShardPtrSlot>` (see host_fs_backed_block_storage.h
    /// / memory/include/gpu_slot_pool.h) sized to `kGpuFileMaxShards`
    /// (4) pointers regardless of `num_shards` -- only the first
    /// `num_shards` entries are ever populated/read; the rest stay
    /// null.  Element i carries the same value as `d_shards_host[i]`.
    /// Populated by `acquire_device_handle` via an ASYNC H2D copy
    /// queued on the caller's stream (no synchronous cudaMalloc on
    /// the hot path -- the pool's backing memory is allocated once,
    /// at first use per cuda device); released -- also asynchronously,
    /// stream-ordered -- by `release_device_handle`.
    ///
    /// Why both `d_shards_host` and `d_shards_dev`?
    ///   - `d_shards_host` is the source of truth and the canonical
    ///     iterator surface for host code (lifetime / cleanup loops
    ///     in `release_device_handle`).
    ///   - `d_shards_dev` exists so io_engine kernels can do their
    ///     own stripe selection on the GPU --
    ///     `dh = entries[tid].shards[fd_idx]` -- exactly mirroring
    ///     legacy `nvme_batch_xfer_kernel`'s indexing into
    ///     `NVMeFilesSpan`. Pre-resolving the shard on the host
    ///     would force one entry per (sub-slice, shard) and lose
    ///     the GPU-parallel mod/div advantage.
    ///
    /// Kernels MUST NOT mutate this array. The pointer itself is
    /// stable for the lifetime of the GpuFileHandle.
    NvmeFileDeviceHandle** d_shards_dev = nullptr;
};

class IBlockStorage {
public:
    virtual ~IBlockStorage() = default;

    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    /// Bind to an already-bootstrapped INvmeStorage and reload the
    /// GpuFile directory from each device's gpu_file_log.bin.
    ///
    /// `storage` MUST outlive this IBlockStorage and MUST already
    /// have called bootstrap(devices) for every device passed here.
    /// `devices` is the subset that may host GpuFile shards; v0.1
    /// behavior is "every device hosts a copy of the gpu_file_log".
    ///
    /// Returns false on any failure; partial state rolled back.
    virtual bool bootstrap(INvmeStorage*                       storage,
                           const std::vector<const Device*>&   devices) = 0;

    /// Close all open GpuFiles + persist + flush.  Safe to call
    /// twice.  Returns false on best-effort partial failure.
    virtual bool shutdown() = 0;

    // ------------------------------------------------------------------
    // Directory
    // ------------------------------------------------------------------

    /// POSIX open(2)-shaped: open an existing GpuFile by
    /// `spec.name`, or (with `GPU_FILE_OPEN_CREATE`) create it if it
    /// doesn't exist yet.
    ///
    ///   flags == GPU_FILE_OPEN_EXISTING (default): open by
    ///            `spec.name` alone -- every other GpuFileSpec field
    ///            is ignored.  Opens every shard's NvmeFile (host fd
    ///            reopened) via INvmeStorage::open_file.  Returns
    ///            nullptr if not found.
    ///
    ///   flags & GPU_FILE_OPEN_CREATE: if `spec.name` doesn't exist,
    ///            allocates one shard NvmeFile per entry in
    ///            `spec.shard_placement` using the underlying
    ///            INvmeStorage (`spec.tensor_shape` consistency is
    ///            checked before any nvme_storage state is touched).
    ///            If it ALREADY exists, behaves like EXISTING (the
    ///            rest of `spec` is ignored) UNLESS
    ///            `GPU_FILE_OPEN_EXCL` is also set, in which case
    ///            that's a failure (nullptr), mirroring
    ///            O_CREAT|O_EXCL.
    ///
    /// Bulk-init knob (only meaningful with CREATE; default when
    /// unset = single-file durable):
    ///   GPU_FILE_OPEN_NO_PERSIST  defers the gpu_file_log rewrite
    ///            AND passes the NVME_OPEN_NO_PERSIST|NVME_OPEN_NO_SYNC
    ///            bulk-init flags through to every underlying
    ///            open_file(CREATE).  Caller MUST follow up with
    ///            flush_metadata() so both layers' logs and the host
    ///            fs reach the platter.  Crash without that flush
    ///            leaves ghost shards which bootstrap reconcile will
    ///            unlink.
    ///
    /// Returns nullptr on:
    ///   - not-found (EXISTING)
    ///   - already-exists (CREATE|EXCL)
    ///   - invalid spec (tensor_shape inconsistency, shard count
    ///     out of range, duplicate device in shard_placement, etc.)
    ///   - any underlying open_file failure (rolls back all
    ///     already-created shards)
    ///
    /// Bulk-init pattern for LMCache (millions of KV blocks) --
    /// note this pattern doesn't keep the bulk-created GpuFiles
    /// open (see close_gpu_file's doc comment on why that matters
    /// for the handle metadata cache):
    /// @code
    /// for (auto& s : specs) {
    ///     GpuFile* gf = bs->open_gpu_file(s, GPU_FILE_OPEN_CREATE |
    ///                                        GPU_FILE_OPEN_NO_PERSIST);
    ///     bs->close_gpu_file(gf);
    /// }
    /// bs->flush_metadata();
    /// @endcode
    virtual GpuFile* open_gpu_file(const GpuFileSpec& spec,
                                   uint32_t flags = GPU_FILE_OPEN_EXISTING) = 0;

    /// Batch create: creates `count` GpuFiles concurrently (multi-
    /// threaded; FS operations run without the global mutex so they
    /// genuinely parallelize at the NVMe level via R11.5's split-lock).
    /// `flags` is typically GPU_FILE_OPEN_CREATE | GPU_FILE_OPEN_NO_PERSIST.
    /// out[i] corresponds to specs[i]; nullptr on per-file failure.
    /// Caller MUST call flush_metadata() once after the batch.
    virtual std::vector<GpuFile*> open_gpu_files_batch(
        const GpuFileSpec* specs, uint32_t count,
        uint32_t flags = GPU_FILE_OPEN_CREATE) = 0;

    /// Close the GpuFile: closes every shard via
    /// INvmeStorage::close_file, persists the gpu_file_log on
    /// every device that hosts a shard.  The GpuFile* remains
    /// valid for re-open via list_gpu_file_names + open_gpu_file
    /// but its shards' host_fds become < 0.
    virtual bool close_gpu_file(GpuFile* file) = 0;

    /// Remove the GpuFile from the directory + delete every shard.
    ///
    /// `persist_now` (default true) controls whether each device's
    /// gpu_file_log is rewritten before this call returns.  Set
    /// false for bulk deletion + finish with flush_metadata().
    /// The on-disk shard `.bin` files are unlinked synchronously
    /// regardless.
    virtual bool delete_gpu_file(GpuFile* file,
                                 bool persist_now = true) = 0;

    /// Batch delete: flattens every file's shards into one
    /// INvmeStorage::delete_files_batch call per participating
    /// device (threaded there), then does the serial GpuFile-log
    /// bookkeeping.  Always defers persist (bulk mode); caller MUST
    /// call flush_metadata() once after the batch.  out_ok[i]
    /// corresponds to files[i]; returns false if any file failed
    /// (out_ok still filled for every i).
    virtual bool delete_gpu_files_batch(GpuFile* const* files,
                                        uint32_t         count,
                                        bool*            out_ok) = 0;

    /// Flush all deferred metadata across all participating
    /// devices: rewrites every dirty gpu_file_log.bin AND calls
    /// INvmeStorage::flush_metadata on every device so any
    /// deferred shard creates / deletes also land.  Safe to call
    /// repeatedly.  Returns false if any sub-flush fails (state
    /// stays dirty; retry is OK).
    virtual bool flush_metadata() = 0;

    /// All GpuFile names known to the directory across all
    /// participating devices.  No host_fd is opened by this call.
    virtual std::vector<std::string> list_gpu_file_names() const = 0;

    /// All currently-OPEN GpuFiles.  Same semantics as
    /// INvmeStorage::list_files: only entries with at least one
    /// open shard show up; use list_gpu_file_names for the
    /// directory-wide view.
    virtual std::vector<GpuFile*> list_open_gpu_files() const = 0;

    // ------------------------------------------------------------------
    // GPU acquire (R6)
    //
    // These DO NOT open or create the GpuFile -- the file must
    // already be in the directory.  They produce a host-side
    // "acquired" handle that bundles per-shard NvmeFileDeviceHandle*
    // and a small GPU-resident pointer table (d_shards_dev).
    //
    // Naming intentionally avoids "open" so this isn't confused
    // with the directory-level open_gpu_file.
    //
    // Semantics:
    //   acquire_device_handle   for each shard NvmeFile, calls the
    //                           underlying INvmeStorage's
    //                           acquire_device_handle; cudaMallocs
    //                           a pointer table of size num_shards
    //                           on the file's GPU and copies the
    //                           per-shard handle pointers into it.
    //
    //                           Returns nullptr if any underlying
    //                           acquire fails (with rollback of the
    //                           ones already done).
    //
    //   release_device_handle   calls
    //                           INvmeStorage::release_device_handle
    //                           on every shard handle, cudaFrees
    //                           d_shards_dev, deletes the
    //                           GpuFileHandle.  No-op on nullptr.
    //                           Idempotent.
    //
    // Lifetime: the returned handle is valid as long as every
    // shard's NvmeFile is alive AND its device's queue_group is
    // alive.  In practice: don't outlive close_gpu_file or
    // INvmeStorage::shutdown.
    // ------------------------------------------------------------------
    // GPU acquire (R6; async pooled R11)
    //
    // These DO NOT open or create the GpuFile -- the file must
    // already be in the directory.  They produce a host-side
    // "acquired" handle that bundles per-shard NvmeFileDeviceHandle*
    // and a small GPU-resident pointer table (d_shards_dev).
    //
    // Naming intentionally avoids "open" so this isn't confused
    // with the directory-level open_gpu_file.
    //
    // configure_handle_pool  MUST be called once, after bootstrap()
    //                        and before the first acquire_device_handle,
    //                        to size this layer's own
    //                        GpuSlotPool<ShardPtrSlot> (the
    //                        d_shards_dev pointer table, sized to
    //                        `l1_capacity` GpuFiles -- this layer has
    //                        no L2 tier of its own; ShardPtrSlot is
    //                        cheap enough to always rebuild on
    //                        demand, see host_fs_backed_block_storage.h)
    //                        AND to forward capacities to the
    //                        underlying INvmeStorage's
    //                        configure_handle_pool (each shard's
    //                        two-tier NvmeFileDeviceHandle cache).
    //                        Unit mismatch note: `l1_capacity` /
    //                        `l2_capacity` here count GpuFiles, but
    //                        INvmeStorage's cache counts individual
    //                        shard NvmeFiles -- this layer forwards
    //                        `l1_capacity * kGpuFileMaxShards` /
    //                        `l2_capacity * kGpuFileMaxShards`
    //                        (worst case: every GpuFile uses the max
    //                        shard count) so a batch of up to
    //                        `l1_capacity` GpuFiles can always have
    //                        every shard resident simultaneously.
    //
    // Semantics (R11.3):
    //   acquire_device_handle   idempotent "ensure resident": for
    //                           each shard NvmeFile, calls the
    //                           underlying INvmeStorage's
    //                           acquire_device_handle (near-free on a
    //                           cache hit; queues an H2D copy only on
    //                           a miss); then resolves/refreshes this
    //                           GpuFile's own ShardPtrSlot (this
    //                           layer's single-tier pool -- always
    //                           rebuilds the slot's content, since a
    //                           shard's underlying GPU pointer may
    //                           have moved since the last call).
    //                           Returns immediately; every field is
    //                           only guaranteed valid for GPU work
    //                           queued on `stream` AFTER this call
    //                           (stream-ordering).
    //
    //                           Returns nullptr if any underlying
    //                           acquire fails (with rollback of the
    //                           ones already done) or this layer's
    //                           own pool is exhausted.
    //
    //   release_device_handle   advisory "may downgrade now" hint --
    //                           calls INvmeStorage::release_device_handle
    //                           (also advisory) on every shard, and
    //                           lets this layer's own ShardPtrSlot be
    //                           LRU-evicted sooner.  Deletes the
    //                           GpuFileHandle struct itself
    //                           immediately (it's plain host memory,
    //                           not GPU-resident).  No-op on nullptr.
    //                           Idempotent.
    //
    // Lifetime: the returned handle is valid as long as every
    // shard's NvmeFile is alive AND its device's queue_group is
    // alive.  In practice: don't outlive close_gpu_file or
    // INvmeStorage::shutdown.
    // ------------------------------------------------------------------

    virtual bool configure_handle_pool(uint32_t l1_capacity, uint32_t l2_capacity) = 0;

    virtual GpuFileHandle* acquire_device_handle (GpuFile* file, GpuStreamHandle stream) = 0;
    virtual void           release_device_handle(GpuFileHandle* handle, GpuStreamHandle stream) = 0;

    /// Batch variant: ensures every GpuFile in files[0..count) has a
    /// resolved handle, doing ONE flattened INvmeStorage batch
    /// acquire across every shard of every file instead of `count`
    /// separate ones -- see nvme_storage.h's
    /// acquire_device_handles_batch doc for why that matters when a
    /// single caller (e.g. a KV-cache adapter resolving a whole
    /// batch's distinct blocks) touches many files at once.
    /// out_handles[i] corresponds to files[i]; each is heap-allocated
    /// exactly like acquire_device_handle's return and must be freed
    /// the same way (release_device_handle, or the caller's own
    /// equivalent bookkeeping).
    ///
    /// Hard requirement: `count` MUST be <= this layer's own
    /// l1_capacity (the ShardPtrSlot pool, in GpuFile units) -- every
    /// file resolved by one batch call must stay resident for the
    /// duration of the call, mirroring INvmeStorage's own batch
    /// contract. Returns false (whole batch failed) on any per-file
    /// failure; `out_handles` is left partially filled past the
    /// failure point.
    virtual bool acquire_device_handles_batch(GpuFile* const* files,
                                              uint32_t         count,
                                              GpuStreamHandle  stream,
                                              GpuFileHandle**  out_handles) = 0;
};

} // namespace tutti

#endif // __TUTTI_BLOCK_STORAGE_BLOCK_STORAGE_H__
