#ifndef __TUTTI_NVME_STORAGE_NVME_FILE_H__
#define __TUTTI_NVME_STORAGE_NVME_FILE_H__

/**
 * nvme_file.h -- the runtime-side NvmeFile handle (metadata-only).
 *
 * Layer: nvme_storage.  This is what callers (block_storage, smokes)
 * hold after a successful create_file / open_file.
 *
 * R11.5: NvmeFile is now metadata-only -- it does NOT hold a host fd.
 * The file's physical LBA layout (extents) is read once at create
 * time via FIEMAP and cached here permanently (extents don't change
 * on ext4 after fallocate).  Host-side IO (read_blocking / write_
 * blocking / sync) opens a temporary fd via `refs_path` on demand
 * and closes it immediately after -- see host_fs_backed_nvme_storage.cpp.
 *
 * The on-disk file no longer has a 4 KiB header prefix (R11.5 removed
 * pwrite(header)); user data starts at byte 0.  All metadata lives in
 * PersistentFileLog, not in-band.
 *
 * A hardlink under `.tutti/.refs/<name>.bin` acts as an inode
 * refcount: external `rm` of the original path won't free the inode
 * (nlink stays 1 via the ref), so LBA extents stay valid for any
 * GPU kernel still reading through them.  Only `delete_file` unlinks
 * both the original path and the ref.
 *
 * Lifetime:
 *   - Created and owned by INvmeStorage.
 *   - Destroyed via INvmeStorage::delete_file().
 *   - close_file() is now a no-op (no fd to close, no cache to erase).
 *   - Holders MUST NOT delete; the storage handles that.
 */

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

#include "lba_extent.h"

namespace tutti {

struct Device;

using NvmeFileId = uint64_t;

struct NvmeFile {
    NvmeFileId            id;
    std::string           name;
    uint64_t              size_bytes;        // logical size (no header prefix)
    const Device*         device;
    std::vector<LbaExtent> extents;          // physical LBAs of the host file
    // ---- host-side path bookkeeping (no fd held) ----
    std::string           host_path;         // original path, e.g. "/mnt/.../foo.bin"
    std::string           refs_path;         // hardlink ref path, e.g. "/mnt/.../.refs/foo.bin"
    uint64_t              data_offset = 0;   // always 0 now (no header)
};

} // namespace tutti

#endif // __TUTTI_NVME_STORAGE_NVME_FILE_H__
