#ifndef __TUTTI_BLOCK_STORAGE_HOST_FS_BACKED_BLOCK_STORAGE_H__
#define __TUTTI_BLOCK_STORAGE_HOST_FS_BACKED_BLOCK_STORAGE_H__

/**
 * host_fs_backed_block_storage.h -- v0.1 IBlockStorage impl.
 *
 * Layered on top of an INvmeStorage instance; per-Device GpuFile
 * directory persisted as `<mount>/.tutti/gpu_file_log.bin` using
 * PersistentGpuFileLog (one full copy on every participating
 * Device, generation-arbitrated at bootstrap).
 *
 * v0.1 limitations:
 *   - GPU acquire/release (R6.3) is implemented as a stub that
 *     returns nullptr.  R6.2 only ships host-side directory + IO
 *     plumbing.  Smokes that need GPU submit go via
 *     nvme_storage_gpu_smoke for now.
 *   - shard placement is strictly 1:1 (no two shards on the same
 *     device per file).
 */

#include "block_storage.h"
#include "gpu_slot_pool.h"   // memory/include; GpuSlotPool<ShardPtrSlot>

#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tutti {

class PersistentGpuFileLog;

/// Fixed-size, POD pointer table -- the pooled payload behind
/// `GpuFileHandle::d_shards_dev`.  Always `kGpuFileMaxShards` (4)
/// entries regardless of a given file's actual `num_shards`; unused
/// trailing entries are null and never read (the kernel already
/// knows `num_shards`).  `&slot.ptrs[0]` IS `d_shards_dev`'s value --
/// this struct has no other members, so reinterpreting a
/// `ShardPtrSlot*` as `NvmeFileDeviceHandle**` is safe and exactly
/// what acquire_device_handle does.
struct ShardPtrSlot {
    NvmeFileDeviceHandle* ptrs[kGpuFileMaxShards];
};

class HostFsBackedBlockStorage : public IBlockStorage {
public:
    HostFsBackedBlockStorage();
    ~HostFsBackedBlockStorage() override;

    HostFsBackedBlockStorage(const HostFsBackedBlockStorage&)            = delete;
    HostFsBackedBlockStorage& operator=(const HostFsBackedBlockStorage&) = delete;

    bool bootstrap(INvmeStorage*                       storage,
                   const std::vector<const Device*>&   devices) override;
    bool shutdown() override;

    GpuFile* open_gpu_file  (const GpuFileSpec& spec,
                             uint32_t flags = GPU_FILE_OPEN_EXISTING) override;
    std::vector<GpuFile*> open_gpu_files_batch(
                            const GpuFileSpec* specs, uint32_t count,
                            uint32_t flags = GPU_FILE_OPEN_CREATE) override;
    bool     close_gpu_file (GpuFile* file)         override;
    bool     delete_gpu_file(GpuFile* file,
                             bool persist_now = true) override;
    bool     delete_gpu_files_batch(GpuFile* const* files, uint32_t count,
                                    bool* out_ok) override;
    bool     flush_metadata ()                      override;

    std::vector<std::string> list_gpu_file_names() const override;
    std::vector<GpuFile*>    list_open_gpu_files()  const override;

    GpuFileHandle* acquire_device_handle (GpuFile* file, GpuStreamHandle stream) override;
    void           release_device_handle(GpuFileHandle* handle, GpuStreamHandle stream) override;
    bool           configure_handle_pool(uint32_t l1_capacity, uint32_t l2_capacity) override;
    bool           acquire_device_handles_batch(GpuFile* const* files, uint32_t count,
                                                GpuStreamHandle stream,
                                                GpuFileHandle** out_handles) override;

private:
    // Per-Device state: one full mirror of the directory + the path
    // we read/write.  Every Device passed into bootstrap() gets one.
    struct PerDeviceState {
        const Device*                          device = nullptr;
        std::string                            log_path;     // <mount>/.tutti/gpu_file_log.bin
        std::unique_ptr<PersistentGpuFileLog>  log;
        bool                                   dirty = false; // pending persist
    };

    // ---- helpers ----
    PerDeviceState* find_state(const Device*) const;
    PerDeviceState* find_state(int32_t device_id) const;

    // Reconcile pass at bootstrap (nvme_storage R5a.0 analogue):
    //   - for each entry in the merged log, ensure every shard's
    //     <gpu_name>.s<i> NvmeFile exists on its device; if any is
    //     missing the entry is dropped (tombstone).
    //   - for every NvmeFile whose name matches "*.s<digit>" but
    //     does NOT correspond to a live entry, unlink it (ghost).
    bool reconcile_locked_();

    // Push the freshest log to every device that is staler.
    void redistribute_logs_locked_(uint64_t target_generation);

    // Build the deterministic shard NvmeFile name "<gpu_name>.s<i>".
    static std::string shard_name_(std::string_view gpu_name, uint32_t i);

    // Try to parse "*.s<digit>" -> (gpu_name, shard_idx).  Returns
    // false if the name doesn't match the convention.
    static bool parse_shard_name_(std::string_view nvme_name,
                                  std::string*     out_gpu_name,
                                  uint32_t*        out_shard_idx);

    // Validate spec.tensor_shape vs total_size + shard_placement.
    bool validate_spec_(const GpuFileSpec&) const;

    // CREATE branch: FS operations (open_file per shard) run WITHOUT
    // mtx_ so bulk-create parallelizes; the bookkeeping section (file_id,
    // log, GpuFile) re-acquires mtx_ internally.  Caller must NOT hold mtx_.
    GpuFile* create_gpu_file_impl_(const GpuFileSpec& spec, bool persist_now);

    // Bookkeeping half of create: assumes `shards` are already created
    // and the caller HOLDS mtx_.  Allocates a file_id, records the
    // directory entry on every mirror log, builds the GpuFile.  Shared
    // by the single-file (create_gpu_file_impl_) and batch
    // (open_gpu_files_batch) paths.
    GpuFile* record_gpu_file_nolock_(const GpuFileSpec& spec,
                                     std::vector<NvmeFile*>& shards,
                                     bool persist_now);

    // GPU_FILE_OPEN_EXISTING batch path (see .cpp for the full
    // doc comment): only spec.name is used per entry.  Caller must
    // NOT hold mtx_.
    std::vector<GpuFile*> open_gpu_files_batch_existing_(
        const GpuFileSpec* specs, uint32_t count);

    // Mark every device dirty (called after any successful
    // mutation; flush_metadata clears).
    void mark_all_dirty_locked_();

    // ---- state ----
    mutable std::mutex                              mtx_;
    INvmeStorage*                                   storage_ = nullptr;
    bool                                            is_open_ = false;

    // PerDeviceState in bootstrap order.  Indexed by device_id ->
    // looked up via id_to_state_; iteration goes in vector order.
    std::vector<std::unique_ptr<PerDeviceState>>    states_;

    // Live GpuFile* keyed by file_id.  We own the unique_ptrs.
    std::map<uint32_t, std::unique_ptr<GpuFile>>    files_;

    // ---- GPU-resident shard-pointer-table pool (R11.3: single-tier
    // -- see block_storage.h's GpuFileHandle::d_shards_dev doc and
    // memory/include/gpu_slot_pool.h) ----
    //
    // Unlike nvme_storage's NvmeFileDeviceHandle (R11.3: two-tier --
    // see tiered_handle_cache.h), ShardPtrSlot does NOT get an L2
    // tier here: its content (each shard's NvmeFileDeviceHandle*) is
    // only valid as of the moment it was last written -- the
    // underlying pointer can move if nvme_storage's own cache evicts
    // and re-promotes that shard -- so caching a stale ShardPtrSlot
    // content across calls would risk a dangling pointer.  Instead,
    // acquire_device_handle ALWAYS re-ensures every shard resident
    // (cheap: usually an L1 hit inside nvme_storage's own cache) and
    // ALWAYS rewrites the slot's content, but reuses the same GPU
    // slot (and its cudaMalloc'd memory) across calls for a given
    // GpuFileId via the LRU below -- so the only recurring cost is
    // one small cudaMemcpyAsync (~32 B), never a cudaMalloc.
    //
    // Because there's no L2 pinned backing store to serve as the H2D
    // copy's source (unlike nvme_storage), this layer keeps its own
    // small pinned staging array, index-aligned with the
    // GpuSlotPool's device array (see ShardPool below).
    struct ShardPool {
        std::unique_ptr<GpuSlotPool<ShardPtrSlot>> gpu;
        ShardPtrSlot*                              pinned_staging = nullptr;  // cudaMallocHost'd
        uint32_t                                   capacity       = 0;

        ~ShardPool() { if (pinned_staging != nullptr) cudaFreeHost(pinned_staging); }
    };

    uint32_t                                    handle_pool_capacity_ = 0;   // == l1_capacity
    std::mutex                                  shard_pools_mtx_;
    std::unordered_map<int, std::unique_ptr<ShardPool>> shard_pools_;

    // Resident-slot index + LRU (single tier, this layer's own --
    // separate from nvme_storage's internal LRU).  Guarded by
    // resident_mtx_ (distinct from shard_pools_mtx_ / mtx_ -- this is
    // pure bookkeeping around ShardPtrSlot residency, never held
    // while blocking on anything else).
    std::mutex                                          resident_mtx_;
    std::unordered_map<GpuFileId, ShardPtrSlot*>        resident_slot_;
    std::list<GpuFileId>                                lru_;
    std::unordered_map<GpuFileId, std::list<GpuFileId>::iterator> lru_pos_;

    ShardPool* get_or_init_pool_();

    // Ensures `file->id` has a resident ShardPtrSlot, rewriting its
    // content from `value` (always -- see the ShardPool doc comment
    // above) and evicting the LRU tail (advisory-released, i.e. not
    // itself in-flight elsewhere) if the pool is full and this is a
    // new residency.  `protect`, if non-null, excludes those ids
    // from eviction consideration -- used by
    // acquire_device_handles_batch so resolving file N of a batch
    // never evicts the slot just resolved for file N-1 of the SAME
    // batch (mirrors TieredHandleCache's batch protect-set fix).
    // Returns the device pointer, or nullptr on exhaustion.
    ShardPtrSlot* resolve_shard_slot_(GpuFileId id, const ShardPtrSlot& value,
                                      cudaStream_t stream,
                                      const std::unordered_set<GpuFileId>* protect = nullptr);

    // Advisory: lets `id`'s ShardPtrSlot be evicted from the LRU
    // (frees the GPU slot for reuse) sooner than it otherwise would.
    void release_shard_slot_(GpuFileId id, cudaStream_t stream);
};

} // namespace tutti

#endif // __TUTTI_BLOCK_STORAGE_HOST_FS_BACKED_BLOCK_STORAGE_H__
