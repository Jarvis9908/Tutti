#ifndef __TUTTI_NVME_STORAGE_HOST_FS_BACKED_NVME_STORAGE_H__
#define __TUTTI_NVME_STORAGE_HOST_FS_BACKED_NVME_STORAGE_H__

/**
 * host_fs_backed_nvme_storage.h -- the v0.1 INvmeStorage impl.
 *
 * Layer: nvme_storage.
 *
 * Strategy:
 *   - For every Device passed to bootstrap(), the storage:
 *       1. Picks `<mount_root>/<snvme_minor_dir>` as the host
 *          mount point (e.g. /mnt/tutti/snvme0).
 *       2. If the underlying snvme block device is unformatted,
 *          mkfs.ext4 it (one-shot).
 *       3. mount(2) it read+write.
 *       4. mkdir(2) `<mount>/.tutti/`
 *       5. PersistentFileLog::load_or_init(<mount>/.tutti/file_log.bin)
 *
 *   - create_file:
 *       1. open(<mount>/.tutti/<name>.bin, O_CREAT|O_RDWR)
 *       2. fallocate(size)   [R11.5: no in-band header -- user data
 *          starts at byte 0; all metadata lives in PersistentFileLog]
 *       3. fsync (unless NVME_OPEN_NO_SYNC)
 *       4. read_extents(fd) -> LbaExtent[]
 *       5. linkat(.refs/<name>.bin)  [R11.5: inode refcount so an
 *          external `rm` of the original path doesn't free the inode
 *          while a GPU kernel may still be reading through its LBAs]
 *       6. close(fd)  [R11.5: NvmeFile is metadata-only -- no held fd]
 *       7. PersistentFileLog::add() + persist()
 *       8. return NvmeFile*
 *
 *   - read/write/sync open a TEMPORARY fd via the .refs/ hardlink
 *     (pread/pwrite/fsync) and close it immediately after.  No fd is
 *     held between calls.  `byte_offset` is logical (relative to byte
 *     0 of user data); data_offset is 0 (R11.5: no header prefix).
 *
 *   - close_file: no-op (R11.5: no fd, no cache eviction; the file
 *     stays resident in s->files until delete_file).
 *
 *   - shutdown:
 *       1. syncfs() the mount if dirty (R11.5: no held fds to close).
 *       2. PersistentFileLog::persist() one last time.
 *       3. umount(2) every mount in reverse order.
 *
 * v0.1 limitations:
 *   - Hardcoded ext4 (mkfs.ext4 / mount type=ext4).  Other host
 *     filesystems work in principle (FIEMAP is generic VFS) but
 *     are not exercised here.
 *   - No quota; create_file just lets fallocate spit ENOSPC if
 *     the device fills up.
 *   - No file rename; delete + create.
 *   - No directory hierarchy; flat namespace under .tutti/.
 *   - Single std::mutex around all public methods.  Fine for
 *     bootstrap / smoke; revisit when block_storage starts
 *     hammering.
 */

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "nvme_storage.h"
#include "tiered_handle_cache.h"   // memory/include; TieredHandleCache<NvmeFileDeviceHandle, NvmeFileId>

namespace tutti {

class PersistentFileLog;

class HostFsBackedNvmeStorage : public INvmeStorage {
public:
    struct Config {
        // Where to mkdir per-device mount points.  The storage will
        // mkdir(`<root>/snvme<minor>`) and mount the block device on it.
        std::string mount_root = "/mnt/tutti";
        // Whether to mkfs.ext4 if the underlying block device has no
        // recognised filesystem.  Set false in production deployments
        // where ops pre-formats the disks.
        bool        auto_mkfs = true;
    };

    explicit HostFsBackedNvmeStorage(Config cfg);
    HostFsBackedNvmeStorage();              // uses Config{} defaults
    ~HostFsBackedNvmeStorage() override;

    HostFsBackedNvmeStorage(const HostFsBackedNvmeStorage&)            = delete;
    HostFsBackedNvmeStorage& operator=(const HostFsBackedNvmeStorage&) = delete;

    // ---- INvmeStorage --------------------------------------------------

    bool bootstrap(const std::vector<const Device*>& devices) override;
    bool shutdown() override;

    uint64_t total_capacity   (const Device*) const override;
    uint64_t available_capacity(const Device*) const override;
    std::string mount_path     (const Device*) const override;

    NvmeFile* open_file(const Device*    device,
                        std::string_view name,
                        uint32_t         flags       = NVME_OPEN_EXISTING,
                        uint64_t         create_size = 0) override;
    bool      open_files_batch(const CreateSpec* specs, uint32_t count,
                               uint32_t flags, NvmeFile** out) override;
    bool      flush_metadata(const Device* device)  override;
    bool      close_file(NvmeFile* file) override;
    bool      delete_file(NvmeFile* file, bool persist_now = true) override;
    bool      delete_files_batch(NvmeFile* const* files, uint32_t count,
                                 bool* out_ok) override;
    std::vector<NvmeFile*>   list_files     (const Device*) const override;
    std::vector<std::string> list_file_names(const Device*) const override;

    ssize_t   read_blocking (NvmeFile*, uint64_t off, void*       dst, size_t len) override;
    ssize_t   write_blocking(NvmeFile*, uint64_t off, const void* src, size_t len) override;
    bool      sync(NvmeFile*) override;

    NvmeFileDeviceHandle* acquire_device_handle (NvmeFile* file, GpuStreamHandle stream) override;
    void                  release_device_handle(NvmeFile* file, GpuStreamHandle stream) override;
    bool                  configure_handle_pool(uint32_t l1_capacity, uint32_t l2_capacity) override;
    bool                  acquire_device_handles_batch(NvmeFile* const*      files,
                                                       uint32_t              count,
                                                       GpuStreamHandle       stream,
                                                       NvmeFileDeviceHandle** out_handles) override;
    void                  admit_to_cache(NvmeFile* file) override;
    CacheStats            cache_stats() const override;

private:
    struct PerDeviceState {
        const Device*                       device = nullptr;
        std::string                         snvme_blk_path;   // /dev/snvmeNn1
        std::string                         mount_path;       // e.g. /mnt/tutti/snvme0
        std::unique_ptr<PersistentFileLog>  log;
        // Open NvmeFiles, keyed by file_id.  We own the unique_ptrs.
        std::map<uint64_t, std::unique_ptr<NvmeFile>> files;
        // Did THIS bootstrap() perform the mount(2)?  Used to decide
        // whether shutdown() should umount(2).
        bool                                we_mounted = false;

        // Bulk-init dirty flags (R5a.1):
        //   dirty_unsynced_files   set whenever create_file(sync_now=false)
        //                          accepted bytes/extents into the page
        //                          cache without an fsync.  Cleared by a
        //                          successful syncfs(2) inside
        //                          flush_metadata().
        //   dirty_unpersisted_log  set whenever create_file/delete_file
        //                          mutated the log without a subsequent
        //                          log.persist().  Cleared by a successful
        //                          log.persist() inside flush_metadata().
        // shutdown() also drains both flags (syncfs()s the mount if
        // dirty, and the final s.log->persist() lands any deferred
        // entries).
        bool                                dirty_unsynced_files   = false;
        bool                                dirty_unpersisted_log  = false;
    };

    PerDeviceState* find_state(const Device*);
    const PerDeviceState* find_state(const Device*) const;

    bool mkfs_if_needed_locked(const std::string& blk_path);
    bool mount_if_needed_locked(PerDeviceState& s);
    bool umount_locked(PerDeviceState& s);

    // R11.5: FS operations (open/fallocate/fiemap/linkat) run WITHOUT
    // mtx_ so bulk-create parallelizes; the bookkeeping section (log
    // add, files map insert, dirty flags) re-acquires mtx_ internally.
    // Caller must NOT hold mtx_ when calling this.
    bool create_file_locked(PerDeviceState& s,
                            std::string_view name,
                            uint64_t        size_bytes,
                            bool            persist_now,
                            bool            sync_now,
                            NvmeFile**      out);

    // C0 reconcile (R5a.1): after PersistentFileLog::load_or_init,
    // walk <mount>/.tutti/ to find:
    //   - log entries whose <name>.bin is missing -> drop the entry
    //     (tombstone left by a delete_file that crashed between
    //      ::unlink and log.persist).
    //   - <name>.bin files with no corresponding log entry -> unlink
    //     them (ghost left by a create_file that crashed between
    //      file pwrite/fsync and log.persist).
    // If anything was changed, persist the log once.  Single-pass,
    // best-effort; logs warnings but never fails bootstrap.
    bool reconcile_locked_(PerDeviceState& s);

    // Helpers for snvme path conventions:  /dev/ssnvme<m> -> /dev/snvme<m>n1
    // and minor extraction for mount-point naming.
    static std::string blk_path_from_chrdev(const std::string& chrdev);
    static int         minor_from_chrdev (const std::string& chrdev);

    Config                          cfg_;
    mutable std::mutex              mtx_;
    std::vector<std::unique_ptr<PerDeviceState>> states_;   // device order
    bool                            booted_ = false;

    // ---- GPU-resident overflow-extent bookkeeping (R5b, compact
    // handle sizing) ----
    // acquire_device_handle() cudaMalloc's a small `LbaExtent[]`
    // overflow buffer only for files whose FIEMAP extent count
    // exceeds kNvmeFileDeviceHandleInlineExtents (rare: heavy
    // fragmentation).  Unlike the pooled handle slot itself, overflow
    // buffers are NOT pooled -- they're rare and variably sized, not
    // worth a slab allocator.  Tracked host-side keyed by NvmeFileId
    // (stable) rather than by handle pointer (R11.3: GPU addresses
    // are recycled across L1/L2 tiers and no longer uniquely / stably
    // identify a file), so it survives however many times a file's
    // handle round-trips between L1 and L2.  Freed only when the file
    // itself is erased from the cache (delete_file) via a
    // stream-ordered `cudaLaunchHostFunc`, mirroring the deferred-free
    // pattern the rest of this design uses -- never while a kernel
    // might still be reading through `extents_overflow`.
    std::mutex                                  overflow_mtx_;
    std::unordered_map<NvmeFileId, void*>       overflow_by_file_;

    // ---- two-tier handle METADATA cache (R11.3: CPU-pinned L2
    // backing a small GPU-resident L1 -- see
    // memory/include/tiered_handle_cache.h and
    // doc/tutti_vs_geminifs_rw_and_integration.md) ----
    //
    // "metadata" is deliberate in every name here (metadata_caches_,
    // admit_to_metadata_cache_, ...): this caches the GPU-side handle
    // TEMPLATE (LBA extents, d_qps pointer, block size -- the stuff a
    // kernel needs to *address* the file), NOT the file's data pages.
    // File data lives in the OS page cache / on the NVMe platter and
    // is a completely separate concern; do not conflate the two.
    //
    // One TieredHandleCache<NvmeFileDeviceHandle, NvmeFileId> per cuda
    // device this storage ever acquires a handle on, sized to
    // `l1_capacity_`/`l2_capacity_` (set once via configure_handle_pool,
    // BEFORE the first acquire_device_handle call).  Lazily allocated
    // per device because we don't know which cuda devices exist until
    // a file living on one is actually acquired (the cuda_device comes
    // from that file's Device's queue_group, not from this class's own
    // state).
    uint32_t                                              l1_capacity_ = 0;
    uint32_t                                              l2_capacity_ = 0;
    mutable std::mutex                                    metadata_caches_mtx_;
    std::unordered_map<int, std::unique_ptr<TieredHandleCache<NvmeFileDeviceHandle, NvmeFileId>>> metadata_caches_;

    // Returns the cache for `cuda_device`, lazily init'ing it on
    // first use.  Returns nullptr if configure_handle_pool was never
    // called, or init() fails.
    TieredHandleCache<NvmeFileDeviceHandle, NvmeFileId>* get_or_init_metadata_cache_(int cuda_device);

    // Builds the host-side NvmeFileDeviceHandle template for `file`
    // (walks FIEMAP-derived extents already cached on `file`, reads
    // controller/queue-group state) -- the TieredHandleCache builder.
    // Also builds the (rare) overflow extent buffer if needed and
    // records it in overflow_by_file_.  `stream` is used only for the
    // overflow buffer's async fill; the returned template itself is
    // plain host memory.
    bool build_handle_template_(NvmeFile* file, cudaStream_t stream, NvmeFileDeviceHandle* out);

    // Admit `file`'s handle template into its cuda device's L2 tier
    // (CPU-pinned) WITHOUT promoting to L1 -- see
    // TieredHandleCache::admit.  Called from open_file() so that
    // "present in the cache" tracks "file is open" (close_file /
    // delete_file erase it again).  Best-effort no-op when the handle
    // pool was never configured (pure host-side users doing no GPU IO
    // -- `l1_capacity_ == 0` -- for whom the cache is irrelevant) or
    // the file's Device has no queue_group.  Temporarily sets the
    // file's cuda_device current around the (rare) overflow-buffer
    // allocation so open_file stays callable from any device context.
    void admit_to_metadata_cache_(NvmeFile* file);

    // Fully removes `file` from its cuda device's two-tier cache
    // (both L1 and L2 -- see TieredHandleCache::erase) and frees its
    // overflow buffer if any.  Called from close_file() (the file
    // still exists on disk -- a later open_file rebuilds the template)
    // AND delete_file() (the file is gone) BEFORE the NvmeFile object
    // itself is destroyed / its fd closed.  No-op if this file's cache
    // was never lazily created (i.e. its handle was never acquired
    // and it was never admitted).  Uses the default stream -- neither
    // caller has a stream parameter of its own; this is fine under the
    // ordinary convention that a file's IO is expected to be done
    // before it's closed/deleted (a caller closing/deleting a file
    // whose handle is still being read on another stream is violating
    // that convention, same as any other cross-stream resource-
    // lifetime hazard in CUDA).
    void erase_from_metadata_cache_(NvmeFile* file);
};

} // namespace tutti

#endif // __TUTTI_NVME_STORAGE_HOST_FS_BACKED_NVME_STORAGE_H__
