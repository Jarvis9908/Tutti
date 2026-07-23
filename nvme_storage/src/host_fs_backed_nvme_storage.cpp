/**
 * host_fs_backed_nvme_storage.cpp -- mount-the-snvme-block-device
 * implementation of INvmeStorage.
 */

#include "host_fs_backed_nvme_storage.h"
#include "fiemap_helper.h"
#include "persistent_file_log.h"

#include "../../device_manager/include/local_nvme_device.h"
#include "../../coordinator/include/device.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <fcntl.h>
#include <filesystem>
#include <linux/magic.h>
#include <sys/ioctl.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>

namespace tutti {

namespace {

// Info-level bring-up logging is opt-in: quiet by default so benchmark
// and nsys runs stay readable; set TUTTI_VERBOSE=1 to re-enable.
bool tutti_verbose() {
    static const bool v = std::getenv("TUTTI_VERBOSE") != nullptr;
    return v;
}
#define TUTTI_INFO(...) do { if (tutti_verbose()) std::fprintf(stderr, __VA_ARGS__); } while (0)

constexpr uint32_t kNvmeBlockSize = 4096;   // matches all NVMe deployments
                                            // we currently support; the
                                            // bootstrap path validates against
                                            // LocalNvmeDevice::blk_size.

bool path_exists(const std::string& p) {
    struct stat st{};
    return ::stat(p.c_str(), &st) == 0;
}

bool is_mounted(const std::string& mount_point) {
    // "Is something mounted at mount_point?" -- compare statfs of
    // mount_point and its parent.  If they differ we know there's
    // a mount boundary here.
    struct stat st_self{}, st_parent{};
    if (::stat(mount_point.c_str(), &st_self) != 0) return false;
    std::string parent = mount_point + "/..";
    if (::stat(parent.c_str(),       &st_parent) != 0) return false;
    return st_self.st_dev != st_parent.st_dev;
}

// Run a system command, log the command line, return true on rc=0.
bool run_cmd(const std::string& cmd) {
    TUTTI_INFO("[nvme_storage] $ %s\n", cmd.c_str());
    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::fprintf(stderr,
            "[nvme_storage]   command failed (rc=%d)\n", rc);
        return false;
    }
    return true;
}

// "/dev/ssnvme7" -> 7
int minor_from_chrdev_impl(const std::string& chrdev) {
    static const std::string prefix = "/dev/ssnvme";
    if (chrdev.compare(0, prefix.size(), prefix) != 0) return -1;
    const std::string tail = chrdev.substr(prefix.size());
    if (tail.empty()) return -1;
    char* endp = nullptr;
    long v = std::strtol(tail.c_str(), &endp, 10);
    if (endp == tail.c_str() || v < 0) return -1;
    return (int)v;
}

} // namespace

// ---------------------------------------------------------------------------

std::string HostFsBackedNvmeStorage::blk_path_from_chrdev(
    const std::string& chrdev)
{
    int m = minor_from_chrdev_impl(chrdev);
    if (m < 0) return {};

    // The snvme namespace gendisk is named "snvme<m>n<K>" where <m>
    // matches the chrdev minor and <K> is the kernel's namespace
    // *instance* counter -- NOT the NVMe NSID.  <K> is usually 1, but
    // the kernel bumps it on every rebind of the same controller (e.g.
    // a daemon restart yields snvme0n2 after a bind/unbind/bind cycle),
    // so hardcoding "n1" breaks the moment the device is re-bound.
    // Resolve <K> at runtime via /sys/block, which lists ONLY live
    // gendisks -- this also disambiguates against any stale /dev nodes
    // udev may have left behind from a previous bind.
    char prefix[32];
    std::snprintf(prefix, sizeof(prefix), "snvme%dn", m);
    const std::size_t prefix_len = std::strlen(prefix);

    std::string chosen;
    {
        namespace fs = std::filesystem;
        std::error_code ec;
        for (const auto& de : fs::directory_iterator("/sys/block", ec)) {
            const std::string nm = de.path().filename().string();
            if (nm.compare(0, prefix_len, prefix) != 0) continue;
            // Require "snvme<m>n<digits>" exactly -- reject partitions
            // ("...p1") and any other non-namespace siblings.
            const char* tail = nm.c_str() + prefix_len;
            if (*tail == '\0') continue;
            bool all_digits = true;
            for (const char* c = tail; *c; ++c) {
                if (*c < '0' || *c > '9') { all_digits = false; break; }
            }
            if (!all_digits) continue;
            chosen = nm;
            break;   // exactly one expected for a single-namespace ctrl
        }
    }

    if (chosen.empty()) {
        // No live gendisk found.  Fall back to the legacy n1 name so
        // the downstream error message names a plausible path; this
        // also preserves behaviour if /sys/block is unavailable.
        char buf[64];
        std::snprintf(buf, sizeof(buf), "/dev/snvme%dn1", m);
        std::fprintf(stderr,
            "[nvme_storage] blk_path_from_chrdev: no live /sys/block/%s* "
            "namespace for chrdev %s; falling back to %s\n",
            prefix, chrdev.c_str(), buf);
        return std::string(buf);
    }
    return "/dev/" + chosen;
}

int HostFsBackedNvmeStorage::minor_from_chrdev(const std::string& chrdev) {
    return minor_from_chrdev_impl(chrdev);
}

// ---------------------------------------------------------------------------

HostFsBackedNvmeStorage::HostFsBackedNvmeStorage(Config cfg)
    : cfg_(std::move(cfg))
{}

HostFsBackedNvmeStorage::HostFsBackedNvmeStorage()
    : cfg_(Config())
{}

HostFsBackedNvmeStorage::~HostFsBackedNvmeStorage() {
    // Best-effort shutdown so destructor doesn't leave mounts behind.
    if (booted_) {
        (void)shutdown();
    }
}

// ---------------------------------------------------------------------------
// State lookup
// ---------------------------------------------------------------------------

HostFsBackedNvmeStorage::PerDeviceState*
HostFsBackedNvmeStorage::find_state(const Device* dev) {
    for (auto& sp : states_) {
        if (sp->device == dev) return sp.get();
    }
    return nullptr;
}
const HostFsBackedNvmeStorage::PerDeviceState*
HostFsBackedNvmeStorage::find_state(const Device* dev) const {
    for (const auto& sp : states_) {
        if (sp->device == dev) return sp.get();
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

bool HostFsBackedNvmeStorage::mkfs_if_needed_locked(const std::string& blk_path) {
    // Use blkid to see if the block device already has a recognised fs.
    // We don't link against libblkid; just shell out and check rc.
    char cmd[512];
    std::snprintf(cmd, sizeof(cmd),
        "blkid -p -s TYPE -o value %s 2>/dev/null",
        blk_path.c_str());
    FILE* fp = ::popen(cmd, "r");
    if (fp == nullptr) {
        std::fprintf(stderr,
            "[nvme_storage] popen(blkid) failed: errno %d\n", errno);
        return false;
    }
    char line[64] = {0};
    if (::fgets(line, sizeof(line), fp) != nullptr) {
        // Got something -- already formatted.  Trim newline.
        for (char* c = line; *c; ++c) {
            if (*c == '\n' || *c == '\r') { *c = 0; break; }
        }
        ::pclose(fp);
        TUTTI_INFO(
            "[nvme_storage] %s already formatted (TYPE=%s); reusing\n",
            blk_path.c_str(), line);
        return true;
    }
    ::pclose(fp);

    if (!cfg_.auto_mkfs) {
        std::fprintf(stderr,
            "[nvme_storage] %s has no fs and auto_mkfs=false; aborting\n",
            blk_path.c_str());
        return false;
    }

    // mkfs.ext4 -F (force, accept "device is in use" warnings)
    std::string mkfs_cmd = "mkfs.ext4 -F -q " + blk_path;
    return run_cmd(mkfs_cmd);
}

bool HostFsBackedNvmeStorage::mount_if_needed_locked(PerDeviceState& s) {
    // mkdir mount_path if it doesn't exist
    if (!path_exists(s.mount_path)) {
        std::error_code ec;
        std::filesystem::create_directories(s.mount_path, ec);
        if (ec) {
            std::fprintf(stderr,
                "[nvme_storage] mkdir(%s) failed: %s\n",
                s.mount_path.c_str(), ec.message().c_str());
            return false;
        }
    }

    if (is_mounted(s.mount_path)) {
        TUTTI_INFO(
            "[nvme_storage] %s is already a mount point; reusing\n",
            s.mount_path.c_str());
        s.we_mounted = false;
        return true;
    }

    // mount(2) with type ext4, no special flags.
    if (::mount(s.snvme_blk_path.c_str(), s.mount_path.c_str(),
                "ext4", 0, nullptr) != 0) {
        std::fprintf(stderr,
            "[nvme_storage] mount(%s -> %s, ext4) failed: errno %d (%s)\n",
            s.snvme_blk_path.c_str(), s.mount_path.c_str(),
            errno, std::strerror(errno));
        return false;
    }
    s.we_mounted = true;
    TUTTI_INFO(
        "[nvme_storage] mounted %s -> %s\n",
        s.snvme_blk_path.c_str(), s.mount_path.c_str());
    return true;
}

bool HostFsBackedNvmeStorage::umount_locked(PerDeviceState& s) {
    if (!s.we_mounted) return true;
    if (::umount(s.mount_path.c_str()) != 0) {
        std::fprintf(stderr,
            "[nvme_storage] umount(%s) failed: errno %d (%s)\n",
            s.mount_path.c_str(), errno, std::strerror(errno));
        return false;
    }
    s.we_mounted = false;
    std::fprintf(stderr,
        "[nvme_storage] unmounted %s\n", s.mount_path.c_str());
    return true;
}

// ---------------------------------------------------------------------------
// C0 reconcile
// ---------------------------------------------------------------------------
//
// Background:
//   create_file and delete_file each touch host fs + log in sequence
//   without a transaction.  A crash between the two persistent steps
//   leaves one of:
//
//     ghost     <name>.bin exists on host fs but no log entry
//                (create_file crashed after pwrite/fsync, before
//                 log.persist).
//     tombstone log entry exists but <name>.bin is gone
//                (delete_file crashed after ::unlink, before
//                 log.persist).
//
//   Both are user-visible bugs (ghosts occupy disk silently; tombstones
//   make open_file return an entry whose ::open then fails ENOENT).
//   This routine sweeps both classes once at bootstrap time, before
//   anyone calls create_file / open_file.
//
//   Not crash-safe in itself; if we crash mid-reconcile we just rerun
//   on next bootstrap.  A future intent-log (write-ahead log) around
//   create/delete would close this window entirely.
bool HostFsBackedNvmeStorage::reconcile_locked_(PerDeviceState& s) {
    namespace fs = std::filesystem;

    const std::string tutti_dir = s.mount_path + "/.tutti";

    // 1. Snapshot disk: every <name>.bin under .tutti/ except the
    //    log itself.  We deliberately tolerate stray files (no .bin
    //    extension) -- they belong to neither log nor ghost class.
    std::unordered_map<std::string, std::string> bin_on_disk;  // name -> abs path
    std::error_code ec;
    fs::directory_iterator it(tutti_dir, ec);
    if (ec) {
        std::fprintf(stderr,
            "[nvme_storage] reconcile: directory_iterator(%s) failed: %s\n",
            tutti_dir.c_str(), ec.message().c_str());
        return false;
    }
    for (const auto& de : it) {
        if (!de.is_regular_file()) continue;
        const auto& p   = de.path();
        std::string fn  = p.filename().string();
        if (fn == "file_log.bin" || fn == "file_log.bin.tmp") continue;
        // Sibling layers are allowed to drop their own sidecar logs
        // under the same .tutti/ directory.  block_storage puts
        // gpu_file_log.bin here -- we must not treat it as a ghost
        // NvmeFile, otherwise bootstrap would silently nuke it and
        // every GpuFile entry would disappear on the next start.
        // Whitelist the known sidecars explicitly; future layers
        // adding new sidecars must extend this list.
        if (fn == "gpu_file_log.bin"     ||
            fn == "gpu_file_log.bin.tmp") continue;
        // Only consider names ending in .bin.
        constexpr std::string_view kBinExt = ".bin";
        if (fn.size() <= kBinExt.size()) continue;
        if (fn.compare(fn.size() - kBinExt.size(),
                       kBinExt.size(), kBinExt) != 0) continue;
        std::string nm = fn.substr(0, fn.size() - kBinExt.size());
        bin_on_disk.emplace(std::move(nm), p.string());
    }

    // 2. Snapshot log: build a name set first so the iteration order
    //    isn't perturbed by the remove() that follows.
    std::unordered_map<std::string, uint64_t> log_by_name;  // name -> file_id
    log_by_name.reserve(s.log->size());
    for (const auto& e : s.log->entries()) {
        log_by_name.emplace(e.name, e.file_id);
    }

    // 3. Tombstone sweep: log entries whose .bin is missing.
    std::size_t n_tombstone = 0;
    for (const auto& [nm, fid] : log_by_name) {
        if (bin_on_disk.find(nm) != bin_on_disk.end()) continue;
        if (s.log->remove(fid)) {
            ++n_tombstone;
            std::fprintf(stderr,
                "[nvme_storage] reconcile: dropping tombstone "
                "name='%s' file_id=%lu (no .bin on disk)\n",
                nm.c_str(), (unsigned long)fid);
        }
    }

    // 4. Ghost sweep: .bin files with no log entry.
    std::size_t n_ghost = 0;
    for (const auto& [nm, abs] : bin_on_disk) {
        if (log_by_name.find(nm) != log_by_name.end()) continue;
        if (::unlink(abs.c_str()) != 0) {
            std::fprintf(stderr,
                "[nvme_storage] reconcile: unlink(%s) failed: errno %d "
                "(skipping; ghost will be revisited on next bootstrap)\n",
                abs.c_str(), errno);
            continue;
        }
        ++n_ghost;
        std::fprintf(stderr,
            "[nvme_storage] reconcile: removed ghost '%s'\n", abs.c_str());
    }

    // 4b. Dangling ref cleanup: unlink .tutti/.refs/<name>.bin entries
    //     whose name is not in the log (orphans left by a
    //     delete_file that crashed after unlinking the original but
    //     before log.persist, or by an external rm of the original
    //     path that the tombstone sweep above then dropped from the
    //     log).  The .refs/ dir itself is left in place.  Uses the
    //     live log (find_by_name) -- NOT the stale log_by_name snapshot
    //     -- so tombstoned entries are correctly treated as gone.
    {
        const std::string refs_dir = tutti_dir + "/.refs";
        std::error_code ec2;
        fs::directory_iterator rit(refs_dir, ec2);
        if (ec2) {
            // .refs/ may not exist yet (no files created this run);
            // not an error.
        } else {
            for (const auto& de : rit) {
                if (!de.is_regular_file()) continue;
                const auto& p  = de.path();
                std::string fn = p.filename().string();
                constexpr std::string_view kBinExt2 = ".bin";
                if (fn.size() <= kBinExt2.size()) continue;
                if (fn.compare(fn.size() - kBinExt2.size(),
                               kBinExt2.size(), kBinExt2) != 0) continue;
                std::string nm2 = fn.substr(0, fn.size() - kBinExt2.size());
                if (s.log->find_by_name(nm2) != nullptr) continue;  // still live
                if (::unlink(p.c_str()) != 0) {
                    std::fprintf(stderr,
                        "[nvme_storage] reconcile: unlink(ref %s) failed: "
                        "errno %d (skipping)\n", p.c_str(), errno);
                    continue;
                }
                ++n_ghost;
                std::fprintf(stderr,
                    "[nvme_storage] reconcile: removed orphan ref '%s'\n",
                    p.c_str());
            }
        }
    }

    // 5. Persist log only if we changed anything.  A successful run
    //    with both counts == 0 leaves the on-disk log byte-identical.
    if (n_tombstone != 0) {
        if (!s.log->persist()) {
            std::fprintf(stderr,
                "[nvme_storage] reconcile: log.persist failed; "
                "tombstone removals will reappear on next bootstrap\n");
            return false;
        }
    }
    if (n_tombstone != 0 || n_ghost != 0) {
        std::fprintf(stderr,
            "[nvme_storage] reconcile %s: -%zu tombstone(s) -%zu ghost(s); "
            "%zu live entries\n",
            s.mount_path.c_str(), n_tombstone, n_ghost, s.log->size());
    }
    return true;
}

bool HostFsBackedNvmeStorage::bootstrap(
    const std::vector<const Device*>& devices)
{
    std::lock_guard<std::mutex> lock(mtx_);
    if (booted_) {
        std::fprintf(stderr, "[nvme_storage] bootstrap: already booted\n");
        return true;
    }
    if (devices.empty()) {
        std::fprintf(stderr, "[nvme_storage] bootstrap: empty device list\n");
        return false;
    }

    // mkdir mount_root once
    {
        std::error_code ec;
        std::filesystem::create_directories(cfg_.mount_root, ec);
        if (ec) {
            std::fprintf(stderr,
                "[nvme_storage] mkdir(%s): %s\n",
                cfg_.mount_root.c_str(), ec.message().c_str());
            return false;
        }
    }

    states_.reserve(devices.size());
    for (const Device* dev : devices) {
        if (dev == nullptr || dev->backend_private == nullptr) {
            std::fprintf(stderr, "[nvme_storage] null Device\n");
            goto rollback;
        }
        auto* lnd = static_cast<LocalNvmeDevice*>(dev->backend_private);
        if (lnd->blk_size != kNvmeBlockSize) {
            std::fprintf(stderr,
                "[nvme_storage] device %d blk_size=%u != expected %u\n",
                dev->device_id, lnd->blk_size, kNvmeBlockSize);
            goto rollback;
        }

        auto sp = std::make_unique<PerDeviceState>();
        sp->device         = dev;
        sp->snvme_blk_path = blk_path_from_chrdev(lnd->snvme_dev_path);
        if (sp->snvme_blk_path.empty()) {
            std::fprintf(stderr,
                "[nvme_storage] cannot derive blk path from %s\n",
                lnd->snvme_dev_path.c_str());
            goto rollback;
        }
        // Mount point: <mount_root>/snvme<minor>
        int m = minor_from_chrdev(lnd->snvme_dev_path);
        char tail[64];
        std::snprintf(tail, sizeof(tail), "/snvme%d", m);
        sp->mount_path = cfg_.mount_root + tail;

        // 1. mkfs (if needed)
        if (!mkfs_if_needed_locked(sp->snvme_blk_path)) {
            goto rollback;
        }
        // 2. mount
        if (!mount_if_needed_locked(*sp)) {
            goto rollback;
        }
        // 3. mkdir <mount>/.tutti
        std::string tutti_dir = sp->mount_path + "/.tutti";
        {
            std::error_code ec;
            std::filesystem::create_directories(tutti_dir, ec);
            if (ec) {
                std::fprintf(stderr,
                    "[nvme_storage] mkdir(%s): %s\n",
                    tutti_dir.c_str(), ec.message().c_str());
                (void)umount_locked(*sp);
                goto rollback;
            }
        }
        // 4. load file_log.bin
        sp->log = std::make_unique<PersistentFileLog>();
        std::string log_path = tutti_dir + "/file_log.bin";
        if (!sp->log->load_or_init(log_path)) {
            std::fprintf(stderr,
                "[nvme_storage] failed to load %s\n", log_path.c_str());
            (void)umount_locked(*sp);
            goto rollback;
        }

        // 5. C0 reconcile: drop tombstone entries (log says yes, .bin
        //    missing) + unlink ghost .bin (file says yes, log no).
        //    Best-effort -- never fails bootstrap.
        (void)reconcile_locked_(*sp);

        TUTTI_INFO(
            "[nvme_storage] device %d ready: blk=%s mount=%s entries=%zu\n",
            dev->device_id, sp->snvme_blk_path.c_str(),
            sp->mount_path.c_str(), sp->log->size());

        states_.push_back(std::move(sp));
    }

    booted_ = true;
    return true;

rollback:
    for (auto it = states_.rbegin(); it != states_.rend(); ++it) {
        (void)umount_locked(**it);
    }
    states_.clear();
    return false;
}

// ---------------------------------------------------------------------------
// Shutdown
// ---------------------------------------------------------------------------

bool HostFsBackedNvmeStorage::shutdown() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!booted_) return true;
    bool all_ok = true;

    // Reverse-order teardown.
    for (auto it = states_.rbegin(); it != states_.rend(); ++it) {
        PerDeviceState& s = **it;

        // No held fds to close.  Data IO is O_DIRECT / GPU-
        // direct (no page cache); umount(2) itself syncs the
        // filesystem, flushing any fallocate metadata left by the
        // NO_SYNC bulk-init path.  flush_metadata() is the explicit
        // syncfs point if durability is needed before shutdown.

        // Persist log one last time.
        if (s.log && !s.log->persist()) {
            all_ok = false;
        }

        // umount.
        if (!umount_locked(s)) {
            all_ok = false;
        }
    }

    states_.clear();
    booted_ = false;
    return all_ok;
}

// ---------------------------------------------------------------------------
// Capacity
// ---------------------------------------------------------------------------

uint64_t HostFsBackedNvmeStorage::total_capacity(const Device* dev) const {
    std::lock_guard<std::mutex> lock(mtx_);
    const auto* s = find_state(dev);
    if (s == nullptr) return 0;
    struct statvfs sv{};
    if (::statvfs(s->mount_path.c_str(), &sv) != 0) return 0;
    return (uint64_t)sv.f_blocks * (uint64_t)sv.f_frsize;
}

uint64_t HostFsBackedNvmeStorage::available_capacity(const Device* dev) const {
    std::lock_guard<std::mutex> lock(mtx_);
    const auto* s = find_state(dev);
    if (s == nullptr) return 0;
    struct statvfs sv{};
    if (::statvfs(s->mount_path.c_str(), &sv) != 0) return 0;
    return (uint64_t)sv.f_bavail * (uint64_t)sv.f_frsize;
}

std::string HostFsBackedNvmeStorage::mount_path(const Device* dev) const {
    std::lock_guard<std::mutex> lock(mtx_);
    const auto* s = find_state(dev);
    if (s == nullptr) return std::string{};
    return s->mount_path;
}

// ---------------------------------------------------------------------------
// Directory operations
// ---------------------------------------------------------------------------

bool HostFsBackedNvmeStorage::create_file_locked(
    PerDeviceState& s,
    std::string_view name,
    uint64_t       size_bytes,
    bool           persist_now,
    bool           sync_now,
    NvmeFile**     out)
{
    if (out == nullptr) return false;
    *out = nullptr;
    if (name.empty() || size_bytes == 0) return false;

    // The sole caller (open_file's NVME_OPEN_CREATE branch) confirms
    // the name is NOT in the log under mtx_ before calling us.  The
    // FS operations below run WITHOUT mtx_ (so bulk-create can
    // parallelize); the bookkeeping section re-acquires mtx_.  Caller
    // must not create the same name from two threads simultaneously.

    // No in-band header.  User data starts at byte 0.
    // All metadata lives in PersistentFileLog (the authoritative
    // source); the on-disk file is pure user data.
    const std::string host_path = s.mount_path + "/.tutti/" +
                                  std::string(name) + ".bin";
    const std::string refs_dir  = s.mount_path + "/.tutti/.refs";

    int fd = ::open(host_path.c_str(),
                     O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (fd < 0) {
        std::fprintf(stderr,
            "[nvme_storage] open(%s) for create: errno %d (%s)\n",
            host_path.c_str(), errno, std::strerror(errno));
        return false;
    }

    if (::fallocate(fd, 0, 0, (off_t)size_bytes) != 0) {
        std::fprintf(stderr,
            "[nvme_storage] fallocate(%s, %llu): errno %d (%s)\n",
            host_path.c_str(), (unsigned long long)size_bytes,
            errno, std::strerror(errno));
        ::close(fd);
        ::unlink(host_path.c_str());
        return false;
    }

    if (sync_now) {
        if (::fsync(fd) != 0) {
            std::fprintf(stderr,
                "[nvme_storage] fsync(%s): errno %d\n", host_path.c_str(), errno);
            ::close(fd);
            ::unlink(host_path.c_str());
            return false;
        }
    }
    // dirty_unsynced_files (for the sync_now=false case) is set in the
    // bookkeeping section below under mtx_, since it is shared state.

    auto fr = read_extents(fd, kNvmeBlockSize);
    if (!fr.ok) {
        std::fprintf(stderr,
            "[nvme_storage] read_extents(%s): %s\n",
            host_path.c_str(), fr.error.c_str());
        ::close(fd);
        ::unlink(host_path.c_str());
        return false;
    }

    // Create a hardlink ref under .tutti/.refs/ so that an
    // external `rm` of the original path does NOT free the inode
    // (nlink stays 1 via the ref).  This guarantees the LBA extents
    // stay valid for any GPU kernel still reading through them --
    // critical for production safety.  Only delete_file unlinks both.
    // Ensure the .refs/ directory exists (created once per mount).
    {
        std::error_code ec;
        std::filesystem::create_directories(refs_dir, ec);
        if (ec) {
            std::fprintf(stderr,
                "[nvme_storage] mkdir(%s): %s\n",
                refs_dir.c_str(), ec.message().c_str());
            ::close(fd);
            ::unlink(host_path.c_str());
            return false;
        }
    }
    const std::string refs_path = refs_dir + "/" + std::string(name) + ".bin";
    if (::linkat(AT_FDCWD, host_path.c_str(),
                 AT_FDCWD, refs_path.c_str(), 0) != 0) {
        std::fprintf(stderr,
            "[nvme_storage] linkat(%s -> %s): errno %d (%s)\n",
            host_path.c_str(), refs_path.c_str(), errno, std::strerror(errno));
        ::close(fd);
        ::unlink(host_path.c_str());
        return false;
    }

    // Close fd immediately -- NvmeFile does not hold it.
    // Host-side IO (read_blocking / write_blocking / sync) opens a
    // temporary fd via refs_path on demand.  This eliminates fd
    // exhaustion at 1M+ file scale.
    ::close(fd);

    // --- Bookkeeping (mutex-protected) ---
    // The FS operations above (open/fallocate/fsync/fiemap/
    // linkat/close) ran WITHOUT mtx_ so bulk-create can parallelize
    // across threads.  Re-acquire the lock only for the in-memory
    // state mutation (log, files map, dirty flags).
    {
        std::lock_guard<std::mutex> lock(mtx_);

        if (!sync_now) {
            // Bulk-init mode: defer durability to flush_metadata().
            // syncfs(2) inside flush_metadata() drains the fallocate'd
            // extent metadata from the page cache.
            s.dirty_unsynced_files = true;
        }

        // NvmeFileId is globally unique: high 16 bits encode device_id,
        // low 48 bits are the per-device monotonic counter.
        const NvmeFileId fid = ((uint64_t)(s.device->device_id & 0xFFFF) << 48)
                               | (s.log->next_file_id() & 0xFFFFFFFFFFFFULL);

        // Add to PersistentFileLog and (optionally) persist.
        PersistentFileLog::Entry e{};
        e.file_id    = fid;
        e.name       = std::string(name);
        e.size_bytes = size_bytes;
        e.extents    = fr.extents;
        if (!s.log->add(std::move(e))) {
            std::fprintf(stderr,
                "[nvme_storage] log.add returned false (race?)\n");
            ::unlink(refs_path.c_str());
            ::unlink(host_path.c_str());
            return false;
        }
        if (persist_now) {
            if (!s.log->persist()) {
                std::fprintf(stderr,
                    "[nvme_storage] log.persist failed\n");
                // Best-effort rollback: remove from log + delete host file.
                s.log->remove(fid);
                ::unlink(refs_path.c_str());
                ::unlink(host_path.c_str());
                return false;
            }
        } else {
            // Bulk-init mode: leave the entry in memory only.
            // flush_metadata() will rewrite the log once for all
            // accumulated additions.
            s.dirty_unpersisted_log = true;
        }

        auto nf = std::make_unique<NvmeFile>();
        nf->id          = fid;
        nf->name        = std::string(name);
        nf->size_bytes  = size_bytes;
        nf->device      = s.device;
        nf->extents     = fr.extents;
        nf->host_path   = host_path;
        nf->refs_path   = refs_path;
        nf->data_offset = 0;   // no header prefix
        NvmeFile* raw = nf.get();
        s.files[fid] = std::move(nf);
        *out = raw;
    }
    return true;
}

bool HostFsBackedNvmeStorage::flush_metadata(const Device* dev)
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto* s = find_state(dev);
    if (s == nullptr) {
        std::fprintf(stderr,
            "[nvme_storage] flush_metadata: device not bootstrapped\n");
        return false;
    }

    bool ok = true;

    // 1. syncfs(2) the mount once if any open_file(NVME_OPEN_NO_SYNC)
    //    happened.  syncfs flushes the entire filesystem to disk in
    //    a single kernel sweep -- O(dirty pages) instead of O(N
    //    files × 2 fsync) -- which is the whole point of bulk init.
    if (s->dirty_unsynced_files) {
        // Use any file descriptor on the mount.  We open the mount
        // point itself so we don't depend on a particular file
        // staying open.  O_DIRECTORY ensures we fail fast if the
        // path got swapped under us.
        int dfd = ::open(s->mount_path.c_str(),
                         O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (dfd < 0) {
            std::fprintf(stderr,
                "[nvme_storage] flush_metadata: open(%s, O_DIRECTORY): "
                "errno %d\n", s->mount_path.c_str(), errno);
            ok = false;
        } else {
            if (::syncfs(dfd) != 0) {
                std::fprintf(stderr,
                    "[nvme_storage] flush_metadata: syncfs(%s): errno %d\n",
                    s->mount_path.c_str(), errno);
                ok = false;
            } else {
                s->dirty_unsynced_files = false;
            }
            ::close(dfd);
        }
    }

    // 2. Rewrite + fsync + rename the log if any open_file(CREATE) or
    //    delete_file came in with NVME_OPEN_NO_PERSIST/persist_now=false.
    if (s->dirty_unpersisted_log) {
        if (!s->log->persist()) {
            std::fprintf(stderr,
                "[nvme_storage] flush_metadata: log.persist failed\n");
            ok = false;
        } else {
            s->dirty_unpersisted_log = false;
        }
    }

    return ok;
}

NvmeFile* HostFsBackedNvmeStorage::open_file(const Device* dev,
                                              std::string_view name,
                                              uint32_t flags,
                                              uint64_t create_size)
{
    std::unique_lock<std::mutex> lock(mtx_);
    auto* s = find_state(dev);
    if (s == nullptr) {
        std::fprintf(stderr, "[nvme_storage] open_file: device not bootstrapped\n");
        return nullptr;
    }

    std::string nm(name);
    const auto* e = s->log->find_by_name(nm);

    if (e == nullptr) {
        // Doesn't exist.
        if (!(flags & NVME_OPEN_CREATE)) return nullptr;   // EXISTING: not found
        NvmeFile* out = nullptr;
        const bool persist_now = !(flags & NVME_OPEN_NO_PERSIST);
        const bool sync_now    = !(flags & NVME_OPEN_NO_SYNC);
        // Release the lock during create so the FS operations
        // (open/fallocate/fiemap/linkat) run without mtx_ -- bulk-create
        // can parallelize across threads.  create_file_locked
        // re-acquires mtx_ internally for the bookkeeping section.
        lock.unlock();
        if (!create_file_locked(*s, name, create_size, persist_now, sync_now, &out))
            return nullptr;
        // Admission into the device-handle cache is lazy -- deferred
        // to the first acquire_device_handle cold miss (it does the same
        // build_handle_template_ there).  Avoids per-create CUDA
        // overhead during bulk init.
        return out;
    }

    // Already exists.
    if ((flags & NVME_OPEN_CREATE) && (flags & NVME_OPEN_EXCL)) {
        std::fprintf(stderr,
            "[nvme_storage] open_file: '%s' already exists (NVME_OPEN_EXCL)\n",
            nm.c_str());
        return nullptr;
    }

    // Already in memory?
    auto it = s->files.find(e->file_id);
    if (it != s->files.end()) {
        // No fd to re-open -- NvmeFile is metadata-only.
        return it->second.get();
    }

    // Not in memory yet; reconstruct NvmeFile from log entry.
    // Do NOT open a host fd -- the file is metadata-only.
    // Host-side IO opens a temporary fd via refs_path on demand.
    std::string host_path = s->mount_path + "/.tutti/" + nm + ".bin";
    std::string refs_path = s->mount_path + "/.tutti/.refs/" + nm + ".bin";
    auto nf = std::make_unique<NvmeFile>();
    nf->id          = e->file_id;
    nf->name        = e->name;
    nf->size_bytes  = e->size_bytes;
    nf->device      = s->device;
    nf->extents     = e->extents;
    nf->host_path   = host_path;
    nf->refs_path   = refs_path;
    nf->data_offset = 0;   // no header prefix
    NvmeFile* raw = nf.get();
    s->files[e->file_id] = std::move(nf);
    return raw;
}

bool HostFsBackedNvmeStorage::open_files_batch(
    const CreateSpec* specs, uint32_t count, uint32_t flags, NvmeFile** out)
{
    if (count == 0) return true;
    if (specs == nullptr || out == nullptr) return false;
    for (uint32_t i = 0; i < count; ++i) out[i] = nullptr;

    // Spawn worker threads calling open_file() concurrently.
    // open_file briefly locks mtx_ to check the log, releases it during
    // the FS operations (fallocate/fiemap/linkat), and re-acquires it
    // only for the in-memory bookkeeping -- so N threads genuinely
    // parallelize at the NVMe level.  This is the library home for the
    // hand-rolled `parallel_create` loop the smokes used to carry.
    //
    // Cap the worker count at kMaxCreateThreads so a huge batch doesn't
    // hog every CPU core (FS ops are IO-bound and spend most time in
    // the kernel, so more threads than this buys little and starves the
    // rest of the process).
    constexpr uint32_t kMaxCreateThreads = 24;
    const uint32_t hw = std::thread::hardware_concurrency();
    uint32_t n_threads = std::min(count, hw ? hw : 8u);
    n_threads = std::min(n_threads, kMaxCreateThreads);
    const uint32_t per_thread = (count + n_threads - 1) / n_threads;
    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    std::atomic<bool> failed{false};

    for (uint32_t t = 0; t < n_threads; ++t) {
        uint32_t start = t * per_thread;
        uint32_t end   = std::min(start + per_thread, count);
        if (start >= end) break;
        threads.emplace_back([&, start, end]() {
            for (uint32_t i = start; i < end; ++i) {
                NvmeFile* nf = open_file(specs[i].device, specs[i].name,
                                         flags, specs[i].size_bytes);
                if (nf == nullptr) {
                    failed.store(true);
                    continue;
                }
                out[i] = nf;
            }
        });
    }
    for (auto& th : threads) th.join();
    return !failed.load();
}

bool HostFsBackedNvmeStorage::close_file(NvmeFile* file) {
    if (file == nullptr) return false;
    std::lock_guard<std::mutex> lock(mtx_);
    auto* s = find_state(file->device);
    if (s == nullptr) return false;

    // NvmeFile is metadata-only (no held fd) and stays resident
    // in s->files until delete_file.  close_file is therefore a no-op:
    //   - no fd to fsync/close,
    //   - no cache entry to evict (the handle template stays admitted
    //     for the file's lifetime; L1 LRU handles working-set eviction
    //     independently, L2 is sized to hold every touched file).
    // Durability of any data written via write_blocking is the caller's
    // responsibility via flush_metadata(device) / sync(file).
    (void)s;
    return true;
}

bool HostFsBackedNvmeStorage::delete_file(NvmeFile* file,
                                           bool      persist_now) {
    if (file == nullptr) return false;
    std::lock_guard<std::mutex> lock(mtx_);
    auto* s = find_state(file->device);
    if (s == nullptr) return false;

    uint64_t fid = file->id;
    std::string host_path = file->host_path;
    std::string refs_path = file->refs_path;

    // Drop any GPU-resident (L1) / CPU-resident (L2) handle
    // cache entry BEFORE the NvmeFile object itself is destroyed
    // below (s->files.erase invalidates `file`).
    erase_from_metadata_cache_(file);

    // No held fd to close.  Unlink BOTH the original path and
    // the .refs/ hardlink so the inode is actually freed (nlink -> 0).
    // Tolerate ENOENT on either (external rm may have beaten us to
    // the original path; the refs may not exist for files created
    // before the .refs/ hardlink scheme was introduced).
    if (::unlink(host_path.c_str()) != 0 && errno != ENOENT) {
        std::fprintf(stderr,
            "[nvme_storage] unlink(%s): errno %d\n",
            host_path.c_str(), errno);
        return false;
    }
    if (!refs_path.empty()) {
        if (::unlink(refs_path.c_str()) != 0 && errno != ENOENT) {
            std::fprintf(stderr,
                "[nvme_storage] unlink(%s): errno %d\n",
                refs_path.c_str(), errno);
            // non-fatal: the original is already gone; the orphaned
            // ref will be swept by reconcile on next bootstrap.
        }
    }

    s->files.erase(fid);
    if (!s->log->remove(fid)) {
        // Already gone from log; not fatal.
    }
    if (persist_now) {
        return s->log->persist();
    }
    // Bulk-delete mode: caller must call flush_metadata(device) to
    // land the log rewrite.  The on-disk .bin is already unlinked
    // synchronously above -- only the log update is deferred.
    s->dirty_unpersisted_log = true;
    return true;
}

bool HostFsBackedNvmeStorage::delete_files_batch(
    NvmeFile* const* files, uint32_t count, bool* out_ok)
{
    if (count == 0) return true;
    if (files == nullptr || out_ok == nullptr) return false;
    for (uint32_t i = 0; i < count; ++i) out_ok[i] = false;

    // Phase 1 (parallel, no mtx_): cache-erase + the two unlink(2)
    // calls per file -- the dominant per-file cost (2 syscalls that
    // hit the kernel's dentry/inode path).  erase_from_metadata_cache_
    // takes its own metadata_caches_mtx_ (independent of mtx_) and is
    // cheap (CUDA host-side bookkeeping, no GPU sync), so it's safe
    // to call from worker threads same as open_file's FS ops are.
    struct PerFile { std::string host_path, refs_path; const Device* device; uint64_t fid; };
    std::vector<PerFile> pf(count);
    for (uint32_t i = 0; i < count; ++i) {
        if (files[i] == nullptr) continue;
        pf[i] = {files[i]->host_path, files[i]->refs_path,
                 files[i]->device, files[i]->id};
    }

    constexpr uint32_t kMaxDeleteThreads = 24;
    const uint32_t hw = std::thread::hardware_concurrency();
    uint32_t n_threads = std::min(count, hw ? hw : 8u);
    n_threads = std::min(n_threads, kMaxDeleteThreads);
    const uint32_t per_thread = (count + n_threads - 1) / n_threads;
    std::vector<std::thread> threads;
    threads.reserve(n_threads);

    for (uint32_t t = 0; t < n_threads; ++t) {
        uint32_t start = t * per_thread;
        uint32_t end   = std::min(start + per_thread, count);
        if (start >= end) break;
        threads.emplace_back([&, start, end]() {
            for (uint32_t i = start; i < end; ++i) {
                if (files[i] == nullptr) continue;
                erase_from_metadata_cache_(files[i]);
                bool ok = (::unlink(pf[i].host_path.c_str()) == 0 || errno == ENOENT);
                if (ok && !pf[i].refs_path.empty()) {
                    // Non-fatal if this unlink fails -- mirrors
                    // delete_file's single-item tolerance.
                    (void)(::unlink(pf[i].refs_path.c_str()) == 0 || errno == ENOENT);
                }
                out_ok[i] = ok;
            }
        });
    }
    for (auto& th : threads) th.join();

    // Phase 2 (serial, under mtx_): in-memory bookkeeping only --
    // files map erase + log->remove.  Always defers persist (bulk
    // mode); caller MUST flush_metadata(device) after this call.
    bool all_ok = true;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        for (uint32_t i = 0; i < count; ++i) {
            if (files[i] == nullptr) { all_ok = false; continue; }
            if (!out_ok[i]) { all_ok = false; continue; }
            auto* s = find_state(pf[i].device);
            if (s == nullptr) { out_ok[i] = false; all_ok = false; continue; }
            s->files.erase(pf[i].fid);
            (void)s->log->remove(pf[i].fid);   // already-gone is not fatal
            s->dirty_unpersisted_log = true;
        }
    }
    return all_ok;
}

std::vector<NvmeFile*>
HostFsBackedNvmeStorage::list_files(const Device* dev) const {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<NvmeFile*> out;
    const auto* s = find_state(dev);
    if (s == nullptr) return out;
    out.reserve(s->files.size());
    for (const auto& [fid, ptr] : s->files) {
        out.push_back(ptr.get());
    }
    return out;
}

std::vector<std::string>
HostFsBackedNvmeStorage::list_file_names(const Device* dev) const {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<std::string> out;
    const auto* s = find_state(dev);
    if (s == nullptr || !s->log) return out;
    const auto& entries = s->log->entries();
    out.reserve(entries.size());
    for (const auto& e : entries) out.push_back(e.name);
    return out;
}

// ---------------------------------------------------------------------------
// Host-side IO
// ---------------------------------------------------------------------------

ssize_t HostFsBackedNvmeStorage::read_blocking(NvmeFile* file,
                                                uint64_t byte_offset,
                                                void* dst, size_t len)
{
    if (file == nullptr || dst == nullptr) {
        errno = EINVAL;
        return -1;
    }
    if (byte_offset + len > file->size_bytes) {
        errno = EINVAL;
        return -1;
    }
    // Open a temporary O_DIRECT fd via refs_path (the durable
    // hardlink) for this IO, then close it.  O_DIRECT bypasses the
    // page cache -- caller MUST provide a block-aligned buffer, offset,
    // and length (EINVAL otherwise).  NvmeFile no longer holds a fd.
    const std::string& path = !file->refs_path.empty()
                                  ? file->refs_path : file->host_path;
    int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (fd < 0) {
        std::fprintf(stderr,
            "[nvme_storage] read_blocking: open(%s): errno %d\n",
            path.c_str(), errno);
        return -1;
    }
    off_t real_off = (off_t)(file->data_offset + byte_offset);
    auto* p = static_cast<uint8_t*>(dst);
    size_t remaining = len;
    while (remaining > 0) {
        ssize_t r = ::pread(fd, p, remaining, real_off);
        if (r < 0) {
            if (errno == EINTR) continue;
            ::close(fd);
            return -1;
        }
        if (r == 0) break;
        p += r;
        remaining -= (size_t)r;
        real_off += r;
    }
    ::close(fd);
    return (ssize_t)(len - remaining);
}

ssize_t HostFsBackedNvmeStorage::write_blocking(NvmeFile* file,
                                                 uint64_t byte_offset,
                                                 const void* src, size_t len)
{
    if (file == nullptr || src == nullptr) {
        errno = EINVAL;
        return -1;
    }
    if (byte_offset + len > file->size_bytes) {
        errno = EINVAL;
        return -1;
    }
    // Open a temporary O_DIRECT fd via refs_path for this IO.
    // O_DIRECT bypasses the page cache (data goes straight to platter);
    // caller MUST provide a block-aligned buffer, offset, and length
    // (EINVAL otherwise).
    const std::string& path = !file->refs_path.empty()
                                  ? file->refs_path : file->host_path;
    int fd = ::open(path.c_str(), O_RDWR | O_DIRECT | O_CLOEXEC);
    if (fd < 0) {
        std::fprintf(stderr,
            "[nvme_storage] write_blocking: open(%s): errno %d\n",
            path.c_str(), errno);
        return -1;
    }
    off_t real_off = (off_t)(file->data_offset + byte_offset);
    const auto* p = static_cast<const uint8_t*>(src);
    size_t remaining = len;
    while (remaining > 0) {
        ssize_t r = ::pwrite(fd, p, remaining, real_off);
        if (r < 0) {
            if (errno == EINTR) continue;
            ::close(fd);
            return -1;
        }
        if (r == 0) break;
        p += r;
        remaining -= (size_t)r;
        real_off += r;
    }
    ::close(fd);
    return (ssize_t)(len - remaining);
}

bool HostFsBackedNvmeStorage::sync(NvmeFile* file) {
    if (file == nullptr) return false;
    // Open a temporary fd via refs_path, fsync, close.
    const std::string& path = !file->refs_path.empty()
                                  ? file->refs_path : file->host_path;
    int fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        std::fprintf(stderr,
            "[nvme_storage] sync: open(%s): errno %d\n",
            path.c_str(), errno);
        return false;
    }
    bool ok = ::fsync(fd) == 0;
    ::close(fd);
    return ok;
}

} // namespace tutti
