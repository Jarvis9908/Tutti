#ifndef __TUTTI_COORDINATOR_COORDINATOR_H__
#define __TUTTI_COORDINATOR_COORDINATOR_H__

/**
 * coordinator.h -- the top-level Tutti orchestrator (R9).
 *
 * Layer: Coordinator.  Owns one instance of every layer below it and
 * sequences their bring-up / tear-down so application code never has
 * to assemble the registry -> nvme_storage -> block_storage -> memory
 * -> io_engine chain by hand (which is exactly what every smoke used
 * to open-code).
 *
 * Lifecycle:
 *   Coordinator c;
 *   c.bootstrap(cfg);                         // one call brings up the stack
 *   GpuFile* gf = c.open_gpu_file(spec, GPU_FILE_OPEN_CREATE);
 *   GpuFileHandle* h = c.acquire_device_handle(gf);
 *   MemoryRegion* tr = c.allocate_device(bytes, DEVICE, cuda_dev);
 *   c.register_tensor({tr->device_ptr, bytes, {}, granularity});
 *   c.submit_batch(tr, h, file_off, true, stream);   // is_read = true
 *   ... use the GPU tensor ...
 *   c.shutdown();
 *
 * Concurrency: bootstrap() / shutdown() are NOT thread-safe and must
 * be sequenced by the caller.  After bootstrap returns true the
 * data-plane passthroughs (create/acquire/register/submit) follow the
 * thread-safety of the underlying subsystems (memory + block_storage
 * are thread-safe; the single shared LocalNvmeIoEngine serialises its
 * scratch -- see io_engine.h).
 *
 * Control-plane handle methods (handle_for / handle_for_batch /
 * drop_cached_handles / delete_gpu_file / sync_file) access this
 * layer's own last_handle_ / gpu_files_by_id_ maps without locking
 * and are therefore NOT thread-safe -- they must be driven from a
 * single control thread.  (R12 A-3 / 隐患-1: the data-plane
 * submit_batch is the only concurrent-safe entry point; the control
 * plane is single-threaded by design, matching the real call pattern
 * of a single scheduler thread orchestrating file lifecycle.)
 *
 * Bring-up mode (R9.6): cfg.mode selects which IDeviceRegistry sits
 * at the top of the chain -- IN_PROCESS (LocalNvmeDirectRegistry,
 * this process owns the NVMe) or SERVICE_CLIENT
 * (NvmeServiceBackedRegistry, attach via a running
 * nvmeservice_daemon).  Everything below the registry is identical
 * across modes.
 */

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "coordinator_config.h"
#include "../../io_engine/include/local_nvme/nvme_batch.h"  // NvmeBatchInputTensor
#include "../../nvme_storage/include/nvme_storage.h"         // INvmeStorage::CacheStats

namespace tutti {

struct Device;
struct GpuFile;
struct GpuFileSpec;
struct GpuFileHandle;
struct MemoryRegion;
struct TensorRegistrationSpec;
enum class MemoryKind : uint32_t;

using GpuFileId = uint32_t;   // mirror of block_storage.h alias

class HostDeviceMemorySubsystem;
class HostFsBackedNvmeStorage;
class HostFsBackedBlockStorage;
class IDeviceRegistry;
class LocalNvmeIoEngine;
class IIoEngine;

class Coordinator {
public:
    Coordinator();
    ~Coordinator();

    Coordinator(const Coordinator&)            = delete;
    Coordinator& operator=(const Coordinator&) = delete;

    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    /// Bring up the whole stack in dependency order.  On any sub-step
    /// failure, rolls back what it brought up and returns false.
    /// Calling twice without an intervening shutdown() is a logic
    /// error (returns false).
    bool bootstrap(const CoordinatorConfig& cfg);

    /// Tear down in reverse order.  Idempotent; safe to call on a
    /// never-booted / already-shutdown Coordinator.
    void shutdown();

    bool is_booted() const { return booted_; }

    // ------------------------------------------------------------------
    // GpuFile lifecycle (block_storage passthrough)
    // ------------------------------------------------------------------

    /// POSIX open(2)-shaped passthrough to
    /// IBlockStorage::open_gpu_file -- see block_storage.h's
    /// GpuFileOpenFlags (GPU_FILE_OPEN_EXISTING == 0 is the default:
    /// open by spec.name alone; GPU_FILE_OPEN_CREATE[|EXCL|NO_PERSIST]
    /// creates it if missing).  `flags` is plain uint32_t rather than
    /// the enum type so this header doesn't have to pull in
    /// block_storage.h just for one enum (mirrors this header's
    /// existing forward-declaration-only convention).
    GpuFile*       open_gpu_file(const GpuFileSpec& spec, uint32_t flags = 0);

    /// Batch create: passthrough to IBlockStorage::open_gpu_files_batch.
    /// Creates `count` GpuFiles concurrently.  out[i] corresponds to
    /// specs[i]; nullptr on per-file failure.  Records each successfully
    /// created GpuFile in the id->GpuFile* directory.  Caller MUST call
    /// flush_all() (or per-device flush_metadata) after the batch.
    std::vector<GpuFile*> open_gpu_files_batch(
        const GpuFileSpec* specs, uint32_t count,
        uint32_t flags = 0);

    /// True iff `id` was created/opened this session and is therefore
    /// resolvable by handle_for/handle_for_batch (queries the same
    /// id -> GpuFile* directory those use internally).  Callers that
    /// want a precise, per-id error BEFORE spending a batch's L1-
    /// capacity check / GPU work should pre-validate with this rather
    /// than relying on handle_for_batch's own (first-failure-wins,
    /// whole-batch) rejection.
    bool           has_gpu_file(GpuFileId id) const;

    /// `stream` is only used to order the async release (R11) of any
    /// handle still cached for this file's id via `handle_for` --
    /// defaults to the legacy (implicit) stream 0 for callers that
    /// never touched the transparent handle cache for this file.
    bool           delete_gpu_file(GpuFile* gf, bool persist_now = true,
                                   cudaStream_t stream = 0);

    /// Batch delete: passthrough to IBlockStorage::delete_gpu_files_batch
    /// (threaded there) -- drops the id -> GpuFile* directory mapping
    /// and releases any cached handle for each successfully-deleted
    /// id (same host-memory-only release as delete_gpu_file, ordered
    /// on `stream`).  Always defers persist; caller MUST call
    /// block()->flush_metadata() once after the batch.  out_ok[i]
    /// corresponds to files[i]; returns false if any file failed.
    bool           delete_gpu_files_batch(GpuFile* const* files, uint32_t count,
                                          bool* out_ok, cudaStream_t stream = 0);

    /// Async, pooled (R11): borrows a slot from block_storage's
    /// GpuSlotPool and queues an async fill on `stream`; returns
    /// immediately.  The returned handle's fields are only guaranteed
    /// valid for GPU work queued on `stream` AFTER this call (ordinary
    /// CUDA stream-ordering -- mirrors legacy GeminiFS's
    /// cudaMemcpyAsync-then-kernel-launch pattern).  `delete_gpu_file`
    /// also releases (async, on `stream`) any handle still cached for
    /// that id via `handle_for` -- so pass the same stream you'll use
    /// for the delete if you want a well-defined ordering there too.
    GpuFileHandle* acquire_device_handle(GpuFile* gf, cudaStream_t stream);
    void           release_device_handle(GpuFileHandle* h, cudaStream_t stream);

    // ------------------------------------------------------------------
    // Transparent handle cache (id-keyed; async pooled R11.3)
    //
    // The runtime -- not the caller -- owns GpuFile device-handle
    // lifetime.  Callers reference a file by its stable, persistent
    // GpuFileId (survives process restart via block_storage's log) and
    // let the Coordinator lazily acquire the device handle on demand.
    // This keeps the top layer free of acquire/release bookkeeping
    // and makes crash recovery a matter of re-open_gpu_file(spec) ->
    // same id -> handle rebuilt on demand.
    //
    // R11.3: GPU-residency caching/eviction is now owned by
    // block_storage / nvme_storage's own two-tier caches (see
    // memory/include/tiered_handle_cache.h), sized by
    // cfg.handle_l1_capacity / handle_l2_capacity.  This means
    // handle_for/handle_for_batch call acquire_device_handle[_batch]
    // AFRESH on every call ("ensure resident", near-free on a hit --
    // see block_storage.h's doc) rather than keeping their own
    // capacity-capped LRU on top.  This layer only tracks the single
    // most-recently-returned host-side GpuFileHandle* per id (the
    // heap struct acquire_device_handle allocates each call) so it
    // can free the PREVIOUS one once a fresh one has been acquired --
    // pure host-memory hygiene, no GPU-residency side effect (freeing
    // it does NOT call block_->release_device_handle, which would
    // incorrectly hint eviction of the very slot the fresh handle
    // just re-filled for the same id).
    //
    // Only files created / opened through this Coordinator this session
    // are resolvable by id (open_gpu_file records the mapping,
    // regardless of whether it created or opened an existing entry).
    //
    // `handle_for`'s returned pointer is only guaranteed valid for GPU
    // work queued on `stream` AFTER the call that produced/refreshed it
    // -- exactly the contract submit_batch already documents for
    // `stream`.  Pass the SAME stream you'll submit IO on so ordinary
    // CUDA stream-ordering gives you correctness with zero CPU-side
    // waiting.
    // ------------------------------------------------------------------

    /// Resolve a GpuFileId to an acquired handle, ensuring it's
    /// GPU-resident (cache hit inside block_storage/nvme_storage is
    /// near-free).  Returns nullptr if the id was never
    /// created/opened this session or the acquire failed.
    GpuFileHandle* handle_for(GpuFileId id, cudaStream_t stream);

    /// Batch variant of handle_for: resolves every id in
    /// ids[0..count) with a SINGLE flattened acquire across all of
    /// their shards (see block_storage.h's acquire_device_handles_batch
    /// doc) instead of `count` separate ones -- the fix for "one IO
    /// batch touches thousands of distinct files."  out[i] corresponds
    /// to ids[i].  `count` MUST be <= the configured L1 capacity (see
    /// cfg.handle_l1_capacity); returns false (whole batch failed) if
    /// that's violated, any id is unknown, or the underlying batch
    /// acquire fails.
    bool handle_for_batch(const GpuFileId* ids, uint32_t count,
                          cudaStream_t stream, GpuFileHandle** out);

    /// The durability point a caller waits on before marking data
    /// durable in its own index.  Two effects, in order:
    ///
    /// 1. If `stream != nullptr`: cudaStreamSynchronize(stream) so
    ///    that GPU kernels submitting NVMe write commands via that
    ///    stream have actually completed (commands submitted + CQ
    ///    polled).  Without this, fsync could run before the writes
    ///    are even submitted -- a crash-consistency hazard.
    /// 2. fsync every shard's host file.  In this architecture the
    ///    data itself bypasses the host page cache (GPU-direct DMA
    ///    and O_DIRECT host IO), so fsync does NOT flush data pages.
    ///    Its real purpose is: (a) flush dirty **metadata** (extent
    ///    maps from fallocate, though these are normally already
    ///    handled by the create/flush_metadata path), and (b)
    ///    trigger an NVMe FLUSH command via the block layer, which
    ///    flushes the controller's volatile write cache (including
    ///    GPU-written data) to persistent media.
    ///
    /// Pass `stream == nullptr` only when you have already
    /// synchronized the GPU yourself (or there are no GPU writes to
    /// this file in flight).
    /// (R12 B-1 / 隐患-2.)
    bool           sync_file(GpuFileId id, cudaStream_t stream = nullptr);

    /// Release every cached device handle (drops this layer's last-
    /// returned-handle bookkeeping AND advisory-hints block_storage
    /// to downgrade each from L1).  Keeps the id -> GpuFile* directory
    /// mapping intact, so a later handle_for() re-acquires.  Useful
    /// under GPU memory pressure and to prove handles rebuild from a
    /// stable id alone.  `stream` orders the async releases; pass a
    /// stream you know has no other pending work still reading
    /// through any cached handle (or synchronize it yourself first).
    void           drop_cached_handles(cudaStream_t stream);

    /// The configured L1 working-set cap, in GpuFiles (see
    /// cfg.handle_l1_capacity).  Adapters use this to pre-check that
    /// a batch's distinct-file working set fits before calling
    /// handle_for_batch, mirroring the max_entries_per_batch contract.
    uint32_t       handle_cache_capacity() const { return handle_l1_capacity_; }

    // ------------------------------------------------------------------
    // Memory (R7 passthrough)
    // ------------------------------------------------------------------

    MemoryRegion* allocate_device(std::size_t size, MemoryKind kind, int device_id);
    MemoryRegion* register_tensor(const TensorRegistrationSpec& spec);
    void          free(MemoryRegion* region);

    /// Number of NvmeBatchEntry a register_tensor'd region flattens to
    /// (== sum of its IO-slice `num_ios`).  Adapters use this to pack
    /// batches under `io_engine()->max_entries_per_batch()` without
    /// pulling in the memory subsystem's libnvm-bearing headers.
    /// Returns 0 if not booted or the region has no IO-slice table.
    uint32_t batch_entry_count(MemoryRegion* region) const;

    // ------------------------------------------------------------------
    // IO submission (R8.3 io_engine)
    // ------------------------------------------------------------------

    /// Submit a multi-tensor batch (uniform direction) and block until
    /// it completes.  Thin wrapper over IIoEngine::submit_batch.
    bool submit_batch(const std::vector<NvmeBatchInputTensor>& inputs,
                      bool                                     is_read,
                      cudaStream_t                             stream = 0);

    /// Single-tensor convenience: one registered tensor against one
    /// GpuFile shard table at `file_byte_offset`.
    bool submit_batch(MemoryRegion*  tensor_region,
                      GpuFileHandle* file_handle,
                      uint64_t       file_byte_offset,
                      bool           is_read,
                      cudaStream_t   stream = 0);

    // ------------------------------------------------------------------
    // Accessors (advanced callers / smokes that still need the raw
    // subsystem handles).  Valid only between bootstrap() and shutdown().
    // ------------------------------------------------------------------

    HostDeviceMemorySubsystem* memory()    const { return mem_.get(); }
    HostFsBackedNvmeStorage*   storage()   const { return storage_.get(); }
    HostFsBackedBlockStorage*  block()     const { return block_.get(); }
    IIoEngine*                 io_engine() const;
    const std::vector<const Device*>& devices() const { return devices_; }

    /// Two-tier handle-cache counters (passthrough to
    /// INvmeStorage::cache_stats), for observability / stress tests.
    INvmeStorage::CacheStats   cache_stats() const;

private:
    bool                                       booted_ = false;
    std::unique_ptr<IDeviceRegistry>           registry_;
    std::vector<const Device*>                 devices_;
    std::unique_ptr<HostFsBackedNvmeStorage>   storage_;
    std::unique_ptr<HostFsBackedBlockStorage>  block_;
    std::unique_ptr<HostDeviceMemorySubsystem> mem_;
    std::unique_ptr<LocalNvmeIoEngine>         engine_;

    // ---- transparent handle cache (id-keyed) ----
    // id -> in-process GpuFile* (recorded on create/open, dropped on
    // delete).  The GpuFile* is owned by block_storage.
    std::unordered_map<GpuFileId, GpuFile*>       gpu_files_by_id_;
    // id -> most-recently-returned host-side GpuFileHandle* (owned by
    // this map; freed -- host memory only, see the doc comment above
    // -- once a fresh one is acquired for the same id, or via
    // drop_cached_handles / delete_gpu_file / shutdown).
    std::unordered_map<GpuFileId, GpuFileHandle*> last_handle_;
    // L1/L2 capacities computed at bootstrap (GpuFile units) -- see
    // coordinator_config.h's handle_l1_capacity / handle_l2_capacity.
    uint32_t                                      handle_l1_capacity_ = 0;
    uint32_t                                      handle_l2_capacity_ = 0;
};

} // namespace tutti

#endif // __TUTTI_COORDINATOR_COORDINATOR_H__
