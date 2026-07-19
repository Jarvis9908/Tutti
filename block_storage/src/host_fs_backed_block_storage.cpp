#include "host_fs_backed_block_storage.h"
#include "persistent_gpu_file_log.h"
#include "nvme_storage.h"
#include "nvme_file.h"
#include "nvme_file_device_handle.h"

// device_manager: we need Device::device_id to translate between the
// log's persisted shard_device_ids and live Device pointers.
#include "device.h"

#include <cuda_runtime.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <set>
#include <unordered_set>
#include <utility>

namespace tutti {

HostFsBackedBlockStorage::HostFsBackedBlockStorage()  = default;

HostFsBackedBlockStorage::~HostFsBackedBlockStorage() {
    (void)shutdown();
}

// ---------------------------------------------------------------------------
// Lookup helpers
// ---------------------------------------------------------------------------

HostFsBackedBlockStorage::PerDeviceState*
HostFsBackedBlockStorage::find_state(const Device* dev) const {
    if (dev == nullptr) return nullptr;
    for (const auto& s : states_) {
        if (s->device == dev) return s.get();
    }
    return nullptr;
}

HostFsBackedBlockStorage::PerDeviceState*
HostFsBackedBlockStorage::find_state(int32_t device_id) const {
    for (const auto& s : states_) {
        if (s->device->device_id == device_id) return s.get();
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Shard naming convention: <gpu_name>.s<i>
//   * cheap to scan (fixed suffix)
//   * sortable by GpuFile name
//   * a stale shard's parent GpuFile is recoverable from the prefix
// ---------------------------------------------------------------------------
std::string
HostFsBackedBlockStorage::shard_name_(std::string_view gpu_name, uint32_t i) {
    std::string out;
    out.reserve(gpu_name.size() + 4);
    out.append(gpu_name);
    out.push_back('.');
    out.push_back('s');
    if (i < 10) {
        out.push_back('0' + (char)i);
    } else {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%u", i);
        out.append(buf);
    }
    return out;
}

bool
HostFsBackedBlockStorage::parse_shard_name_(std::string_view nvme_name,
                                            std::string*     out_gpu_name,
                                            uint32_t*        out_shard_idx) {
    // Find the last ".s" before a numeric suffix.
    if (nvme_name.size() < 3) return false;
    auto dot = nvme_name.rfind(".s");
    if (dot == std::string_view::npos) return false;
    if (dot == 0) return false;                 // no gpu_name
    auto digits = nvme_name.substr(dot + 2);
    if (digits.empty()) return false;
    uint32_t v = 0;
    for (char c : digits) {
        if (c < '0' || c > '9') return false;
        v = v * 10 + (uint32_t)(c - '0');
        if (v >= kGpuFileMaxShards) return false;  // out of range
    }
    if (out_gpu_name)  out_gpu_name->assign(nvme_name.substr(0, dot));
    if (out_shard_idx) *out_shard_idx = v;
    return true;
}

// ---------------------------------------------------------------------------
// Spec validation
// ---------------------------------------------------------------------------
bool HostFsBackedBlockStorage::validate_spec_(const GpuFileSpec& spec) const {
    if (spec.name.empty())             return false;
    if (spec.total_size == 0)          return false;
    const uint32_t ns = spec.tensor_shape[0];
    const uint32_t ny = spec.tensor_shape[1];
    const uint32_t ts = spec.tensor_shape[2];
    if (ns == 0 || ny == 0 || ts == 0) return false;
    if (ns > kGpuFileMaxShards)        return false;
    // total_size must equal ns * ny * ts (within uint64 -- ns/ny/ts
    // are 32-bit so the product fits in 96 bits but we cap at 64).
    const __uint128_t prod = (__uint128_t)ns * (__uint128_t)ny * (__uint128_t)ts;
    if (prod != (__uint128_t)spec.total_size) return false;
    if (spec.shard_placement.size() != ns) return false;
    // 1:1 placement: no duplicate Device.
    std::set<const Device*> seen;
    for (auto* d : spec.shard_placement) {
        if (d == nullptr)             return false;
        if (find_state(d) == nullptr) return false;   // not bootstrapped
        if (!seen.insert(d).second)   return false;   // duplicate
    }
    return true;
}

// ---------------------------------------------------------------------------
// Bootstrap / shutdown
// ---------------------------------------------------------------------------
bool HostFsBackedBlockStorage::bootstrap(
    INvmeStorage*                       storage,
    const std::vector<const Device*>&   devices)
{
    std::lock_guard<std::mutex> lock(mtx_);
    if (is_open_) return true;
    if (storage == nullptr || devices.empty()) return false;

    storage_ = storage;

    // Build PerDeviceState for every participating Device.  Each
    // gets a full mirror of the gpu_file_log; load_or_init treats
    // ENOENT as "fresh empty log" so a brand new device just makes
    // an empty entry table.
    states_.reserve(devices.size());
    for (const auto* d : devices) {
        if (d == nullptr) {
            std::fprintf(stderr,
                "[block_storage] bootstrap: null Device in list\n");
            states_.clear();
            return false;
        }
        std::string mp = storage_->mount_path(d);
        if (mp.empty()) {
            std::fprintf(stderr,
                "[block_storage] bootstrap: storage->mount_path(dev=%d) "
                "empty -- did you call INvmeStorage::bootstrap first?\n",
                d->device_id);
            states_.clear();
            return false;
        }
        // <mount>/.tutti is created by nvme_storage; we just use it.
        std::string log_path = mp + "/.tutti/gpu_file_log.bin";
        auto sp = std::make_unique<PerDeviceState>();
        sp->device   = d;
        sp->log_path = std::move(log_path);
        sp->log      = std::make_unique<PersistentGpuFileLog>();
        if (!sp->log->load_or_init(sp->log_path)) {
            std::fprintf(stderr,
                "[block_storage] bootstrap: load_or_init(%s) failed\n",
                sp->log_path.c_str());
            states_.clear();
            return false;
        }
        states_.push_back(std::move(sp));
    }

    // Cross-device generation arbitration: pick the highest-gen log
    // and rewrite the others so every device ends up byte-identical.
    uint64_t max_gen = 0;
    PerDeviceState* leader = nullptr;
    for (const auto& s : states_) {
        if (s->log->generation() > max_gen) {
            max_gen = s->log->generation();
            leader  = s.get();
        }
    }
    if (leader != nullptr) {
        for (auto& s : states_) {
            if (s.get() == leader) continue;
            if (s->log->generation() < max_gen) {
                std::fprintf(stderr,
                    "[block_storage] bootstrap: device %d log generation "
                    "%lu < leader %lu (dev=%d); pulling leader's view\n",
                    s->device->device_id,
                    (unsigned long)s->log->generation(),
                    (unsigned long)max_gen, leader->device->device_id);
                s->log->overwrite_from(*leader->log);
                s->dirty = true;
            }
        }
    }

    // Reconcile against nvme_storage state (drop tombstones, unlink
    // ghosts).  Best-effort -- never fails bootstrap.
    (void)reconcile_locked_();

    // If reconcile or arbitration produced changes, persist them
    // before declaring open.  This keeps the on-disk view consistent
    // before any caller starts mutating.
    bool any_dirty = false;
    for (auto& s : states_) any_dirty = any_dirty || s->dirty;
    if (any_dirty) {
        for (auto& s : states_) {
            if (!s->dirty) continue;
            if (!s->log->persist()) {
                std::fprintf(stderr,
                    "[block_storage] bootstrap: persist on dev=%d failed; "
                    "will retry on next flush_metadata\n",
                    s->device->device_id);
            } else {
                s->dirty = false;
            }
        }
    }

    is_open_ = true;
    std::fprintf(stderr,
        "[block_storage] bootstrap ready: devices=%zu entries=%zu\n",
        states_.size(),
        states_.empty() ? 0 : states_.front()->log->size());
    return true;
}

bool HostFsBackedBlockStorage::shutdown() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!is_open_) {
        states_.clear();
        files_.clear();
        return true;
    }
    bool ok = true;

    // Close any still-open GpuFiles.  Each GpuFile holds NvmeFile*
    // references that nvme_storage owns; we call close_file for
    // each so the host_fd's get fsync'd + closed.
    for (auto& [fid, gf] : files_) {
        for (auto* nf : gf->shards) {
            if (nf != nullptr) (void)storage_->close_file(nf);
        }
    }
    files_.clear();

    // Final persist on every dirty device.
    for (auto& s : states_) {
        if (s->dirty) {
            if (!s->log->persist()) {
                std::fprintf(stderr,
                    "[block_storage] shutdown: persist on dev=%d failed\n",
                    s->device->device_id);
                ok = false;
            } else {
                s->dirty = false;
            }
        }
    }

    states_.clear();
    storage_ = nullptr;
    is_open_ = false;
    return ok;
}

// ---------------------------------------------------------------------------
// Reconcile (analogous to nvme_storage's bootstrap-time reconcile)
// ---------------------------------------------------------------------------
//
// For the leader log, walk every entry and its shards:
//   * if any shard NvmeFile is missing -> tombstone (drop entry,
//     keep what shards do exist? we just unlink them too -- safer
//     for a "clean directory" semantic).
// Then for every NvmeFile name on every device matching "*.s<i>":
//   * if it doesn't correspond to a live entry -> ghost, unlink it
//     via INvmeStorage::delete_file.
//
// This is best-effort; we log warnings but never fail bootstrap.
bool HostFsBackedBlockStorage::reconcile_locked_() {
    if (states_.empty()) return true;

    // The arbitration step above already overwrote stragglers from
    // the leader, so every device's in-memory log is identical.
    // Walk states_[0]->log as the source of truth.
    auto& leader_log = *states_.front()->log;

    // Build per-device sets of all NvmeFile names ONCE (not per log
    // entry) -- the previous code called storage_->list_file_names()
    // + std::find inside the 4M-entry loop, an O(N²) anti-pattern
    // that made bootstrap with millions of files take hours.
    std::map<int32_t, std::set<std::string>> files_per_dev;
    for (auto& s : states_) {
        auto names = storage_->list_file_names(s->device);
        files_per_dev[s->device->device_id] =
            std::set<std::string>(names.begin(), names.end());
    }

    // Build set of every "live" shard-name on each device, keyed by
    // device_id.  Used later for ghost detection.
    std::map<int32_t, std::set<std::string>> live_shards_per_dev;

    // 1. Tombstone sweep.
    std::vector<uint32_t> doomed_file_ids;
    for (const auto& e : leader_log.entries()) {
        bool entry_ok = true;
        for (uint32_t i = 0; i < e.num_shards; ++i) {
            const int32_t did = e.shard_device_ids[i];
            const std::string sn = shard_name_(e.name, i);
            auto* state = find_state(did);
            if (state == nullptr) {
                std::fprintf(stderr,
                    "[block_storage] reconcile: entry '%s' shard %u "
                    "references unbootstrapped dev=%d -> tombstone\n",
                    e.name.c_str(), i, did);
                entry_ok = false;
                break;
            }
            // Check the shard NvmeFile actually exists on that
            // device's PersistentFileLog (O(log N) set lookup).
            const auto& dev_files = files_per_dev[did];
            if (dev_files.find(sn) == dev_files.end()) {
                std::fprintf(stderr,
                    "[block_storage] reconcile: entry '%s' shard %u "
                    "missing NvmeFile '%s' on dev=%d -> tombstone\n",
                    e.name.c_str(), i, sn.c_str(), did);
                entry_ok = false;
                break;
            }
            live_shards_per_dev[did].insert(sn);
        }
        if (!entry_ok) doomed_file_ids.push_back(e.file_id);
    }
    if (!doomed_file_ids.empty()) {
        for (uint32_t fid : doomed_file_ids) {
            for (auto& s : states_) (void)s->log->remove(fid);
        }
        for (auto& s : states_) s->dirty = true;
    }

    // 2. Ghost sweep: per device, scan NvmeFile names matching
    //    "*.s<i>" and unlink any that aren't in the live set.
    //    Reuse files_per_dev built above (avoids another
    //    list_file_names call per device).
    for (auto& s : states_) {
        const auto& names = files_per_dev[s->device->device_id];
        for (const auto& nm : names) {
            std::string gpu_name;
            uint32_t    si = 0;
            if (!parse_shard_name_(nm, &gpu_name, &si)) continue;
            // Live?
            const auto& live_set = live_shards_per_dev[s->device->device_id];
            if (live_set.find(nm) != live_set.end()) continue;
            // Ghost: unlink via INvmeStorage::delete_file (deferred,
            // we'll flush at end of bootstrap).
            NvmeFile* nf = storage_->open_file(s->device, nm);
            if (nf == nullptr) continue;
            if (storage_->delete_file(nf, /*persist_now=*/false)) {
                std::fprintf(stderr,
                    "[block_storage] reconcile: removed ghost shard "
                    "'%s' on dev=%d\n", nm.c_str(), s->device->device_id);
            }
        }
        // Drain pending nvme_storage log writes from the ghost sweep.
        (void)storage_->flush_metadata(s->device);
    }
    return true;
}

void HostFsBackedBlockStorage::mark_all_dirty_locked_() {
    for (auto& s : states_) s->dirty = true;
}

// ---------------------------------------------------------------------------
// Directory ops
// ---------------------------------------------------------------------------

// CREATE branch of open_gpu_file() -- FS operations
// (open_file per shard) run WITHOUT mtx_ so bulk-create parallelizes
// across threads; the bookkeeping section (file_id, log, GpuFile) re-
// acquires mtx_ internally.  Caller must NOT hold mtx_.
GpuFile* HostFsBackedBlockStorage::create_gpu_file_impl_(
    const GpuFileSpec& spec, bool persist_now)
{
    if (!validate_spec_(spec)) return nullptr;

    const std::string nm{spec.name};
    const uint32_t num_shards     = spec.tensor_shape[0];
    const uint64_t per_shard_size = spec.total_size / num_shards;

    // Allocate shard NvmeFiles using nvme_storage's bulk-init flags
    // when caller deferred, so create + flush at end stays O(1) RPCs
    // per device.  Rollback: any failure unwinds previously created
    // shards via delete_file.
    uint32_t nvme_flags = NVME_OPEN_CREATE;
    if (!persist_now) nvme_flags |= NVME_OPEN_NO_PERSIST | NVME_OPEN_NO_SYNC;

    // ---- FS operations (NO mtx_ held) ----
    // nvme_storage's open_file releases its own mtx_ during FS
    // ops, so multiple threads calling create_gpu_file_impl_ in parallel
    // genuinely parallelize at the kernel level.
    std::vector<NvmeFile*> shards(num_shards, nullptr);
    bool ok = true;
    for (uint32_t i = 0; i < num_shards; ++i) {
        const Device* d = spec.shard_placement[i];
        const std::string sn = shard_name_(nm, i);
        NvmeFile* nf = storage_->open_file(d, sn, nvme_flags, per_shard_size);
        if (nf == nullptr) {
            std::fprintf(stderr,
                "[block_storage] open_gpu_file(CREATE) '%s': shard %u "
                "open_file failed on dev=%d\n",
                nm.c_str(), i, d->device_id);
            ok = false;
            break;
        }
        shards[i] = nf;
    }
    if (!ok) {
        for (uint32_t i = 0; i < num_shards; ++i) {
            if (shards[i] != nullptr) {
                (void)storage_->delete_file(shards[i],
                                            /*persist_now=*/false);
            }
        }
        for (uint32_t i = 0; i < num_shards; ++i) {
            (void)storage_->flush_metadata(spec.shard_placement[i]);
        }
        return nullptr;
    }

    // ---- Bookkeeping (mtx_ held) ----
    std::lock_guard<std::mutex> lock(mtx_);
    return record_gpu_file_nolock_(spec, shards, persist_now);
}

// Bookkeeping half of create: assumes `shards` are already created and
// the caller HOLDS mtx_.  Allocates a file_id, records the directory
// entry on every device's mirror log, and builds the in-memory GpuFile.
// On log failure, rolls back the shards (delete_file) and returns null.
GpuFile* HostFsBackedBlockStorage::record_gpu_file_nolock_(
    const GpuFileSpec& spec, std::vector<NvmeFile*>& shards, bool persist_now)
{
    const std::string nm{spec.name};
    const uint32_t num_shards = spec.tensor_shape[0];

    // Allocate file_id from the leader log; every device's log will
    // get the same id since they are kept identical.
    uint32_t fid = states_.front()->log->next_file_id();

    // Record on every device's mirror log.
    PersistentGpuFileLog::Entry e;
    e.file_id        = fid;
    e.name           = nm;
    e.total_size     = spec.total_size;
    e.tensor_shape[0] = spec.tensor_shape[0];
    e.tensor_shape[1] = spec.tensor_shape[1];
    e.tensor_shape[2] = spec.tensor_shape[2];
    e.num_shards     = num_shards;
    e.shard_device_ids.resize(num_shards);
    for (uint32_t i = 0; i < num_shards; ++i) {
        e.shard_device_ids[i] = spec.shard_placement[i]->device_id;
    }
    for (auto& s : states_) {
        if (!s->log->add(e)) {
            std::fprintf(stderr,
                "[block_storage] open_gpu_file(CREATE) '%s': mirror log "
                "on dev=%d add() rejected (race?)\n",
                nm.c_str(), s->device->device_id);
            // Best-effort rollback.
            for (auto& sx : states_) (void)sx->log->remove(fid);
            for (auto& nf : shards) (void)storage_->delete_file(nf, false);
            for (auto& sx : states_) (void)storage_->flush_metadata(sx->device);
            return nullptr;
        }
        s->dirty = true;
    }

    if (persist_now) {
        for (auto& s : states_) {
            if (!s->log->persist()) {
                std::fprintf(stderr,
                    "[block_storage] open_gpu_file(CREATE) '%s': persist "
                    "on dev=%d failed; flushed nvme_storage data may "
                    "outlive log entry until next flush_metadata\n",
                    nm.c_str(), s->device->device_id);
                // Stay dirty for retry.
                continue;
            }
            s->dirty = false;
        }
    }

    auto gf      = std::make_unique<GpuFile>();
    gf->id           = fid;
    gf->name         = nm;
    gf->total_size   = spec.total_size;
    gf->tensor_shape[0] = spec.tensor_shape[0];
    gf->tensor_shape[1] = spec.tensor_shape[1];
    gf->tensor_shape[2] = spec.tensor_shape[2];
    gf->shards       = std::move(shards);
    GpuFile* raw = gf.get();
    files_[fid] = std::move(gf);
    return raw;
}

GpuFile* HostFsBackedBlockStorage::open_gpu_file(const GpuFileSpec& spec,
                                                 uint32_t flags) {
    std::unique_lock<std::mutex> lock(mtx_);
    if (!is_open_) return nullptr;
    const std::string nm{spec.name};

    // Already open in this process?
    for (auto& [fid, gf] : files_) {
        if (gf->name == nm) {
            if ((flags & GPU_FILE_OPEN_CREATE) && (flags & GPU_FILE_OPEN_EXCL)) {
                std::fprintf(stderr,
                    "[block_storage] open_gpu_file: '%s' already exists "
                    "(GPU_FILE_OPEN_EXCL)\n", nm.c_str());
                return nullptr;
            }
            return gf.get();
        }
    }

    const auto* e = states_.front()->log->find_by_name(nm);
    if (e == nullptr) {
        // Doesn't exist in the directory.
        if (!(flags & GPU_FILE_OPEN_CREATE)) return nullptr;   // EXISTING: not found
        // Release the lock during create so FS operations
        // (open_file per shard) run without mtx_ — bulk-create can
        // parallelize across threads.  create_gpu_file_impl_ re-
        // acquires mtx_ internally for the bookkeeping section.
        lock.unlock();
        return create_gpu_file_impl_(spec, !(flags & GPU_FILE_OPEN_NO_PERSIST));
    }

    // Exists in the directory but not open in this process yet.
    if ((flags & GPU_FILE_OPEN_CREATE) && (flags & GPU_FILE_OPEN_EXCL)) {
        std::fprintf(stderr,
            "[block_storage] open_gpu_file: '%s' already exists "
            "(GPU_FILE_OPEN_EXCL)\n", nm.c_str());
        return nullptr;
    }

    auto gf = std::make_unique<GpuFile>();
    gf->id           = e->file_id;
    gf->name         = e->name;
    gf->total_size   = e->total_size;
    gf->tensor_shape[0] = e->tensor_shape[0];
    gf->tensor_shape[1] = e->tensor_shape[1];
    gf->tensor_shape[2] = e->tensor_shape[2];
    gf->shards.resize(e->num_shards, nullptr);
    for (uint32_t i = 0; i < e->num_shards; ++i) {
        auto* state = find_state(e->shard_device_ids[i]);
        if (state == nullptr) {
            std::fprintf(stderr,
                "[block_storage] open_gpu_file '%s': shard %u dev_id=%d "
                "not bootstrapped\n", nm.c_str(), i, e->shard_device_ids[i]);
            return nullptr;
        }
        const std::string sn = shard_name_(nm, i);
        NvmeFile* nf = storage_->open_file(state->device, sn);
        if (nf == nullptr) {
            std::fprintf(stderr,
                "[block_storage] open_gpu_file '%s': shard %u open '%s' "
                "on dev=%d failed\n", nm.c_str(), i, sn.c_str(),
                state->device->device_id);
            return nullptr;
        }
        gf->shards[i] = nf;
    }

    GpuFile* raw = gf.get();
    files_[e->file_id] = std::move(gf);
    return raw;
}

std::vector<GpuFile*> HostFsBackedBlockStorage::open_gpu_files_batch(
    const GpuFileSpec* specs, uint32_t count, uint32_t flags)
{
    std::vector<GpuFile*> results(count, nullptr);
    if (count == 0 || specs == nullptr) return results;
    if (!is_open_) return results;

    if (!(flags & GPU_FILE_OPEN_CREATE)) {
        return open_gpu_files_batch_existing_(specs, count);
    }

    const bool persist_now = !(flags & GPU_FILE_OPEN_NO_PERSIST);
    uint32_t nvme_flags = NVME_OPEN_CREATE;
    if (!persist_now) nvme_flags |= NVME_OPEN_NO_PERSIST | NVME_OPEN_NO_SYNC;

    // 1. Validate every spec + flatten all shards into ONE nvme batch
    //    create list.  The threading lives in nvme_storage's
    //    open_files_batch -- block_storage does not spawn
    //    its own threads; it just flattens, batch-creates, and does
    //    the serial bookkeeping.
    std::vector<INvmeStorage::CreateSpec> shard_specs;
    std::vector<uint32_t> shard_base(count, 0);   // first shard idx per file
    std::vector<uint32_t> shard_cnt(count, 0);
    shard_specs.reserve((std::size_t)count * kGpuFileMaxShards);
    for (uint32_t i = 0; i < count; ++i) {
        if (!validate_spec_(specs[i])) return results;   // whole batch fails
        const std::string nm{specs[i].name};
        const uint32_t ns = specs[i].tensor_shape[0];
        const uint64_t per_shard = specs[i].total_size / ns;
        shard_base[i] = (uint32_t)shard_specs.size();
        shard_cnt[i]  = ns;
        for (uint32_t s = 0; s < ns; ++s) {
            shard_specs.push_back({specs[i].shard_placement[s],
                                   shard_name_(nm, s), per_shard});
        }
    }

    // 2. Batch-create every shard concurrently (threading inside nvme).
    std::vector<NvmeFile*> shard_out(shard_specs.size(), nullptr);
    (void)storage_->open_files_batch(shard_specs.data(),
                                     (uint32_t)shard_specs.size(),
                                     nvme_flags, shard_out.data());

    // 3. Serial bookkeeping under mtx_ (fast: no FS work here).
    std::lock_guard<std::mutex> lock(mtx_);
    for (uint32_t i = 0; i < count; ++i) {
        // Gather this file's shards; if any failed, roll back its shards.
        std::vector<NvmeFile*> shards(shard_cnt[i], nullptr);
        bool ok = true;
        for (uint32_t s = 0; s < shard_cnt[i]; ++s) {
            NvmeFile* nf = shard_out[shard_base[i] + s];
            if (nf == nullptr) { ok = false; break; }
            shards[s] = nf;
        }
        if (!ok) {
            for (auto* nf : shards)
                if (nf) (void)storage_->delete_file(nf, false);
            results[i] = nullptr;
            continue;
        }
        results[i] = record_gpu_file_nolock_(specs[i], shards, persist_now);
    }
    return results;
}

// GPU_FILE_OPEN_EXISTING batch path: looks each spec.name up in the
// gpu_file_log (spec.name is the only field used -- mirrors
// open_gpu_file's EXISTING branch), then flattens the not-yet-open
// files' shards into one nvme open_files_batch(EXISTING) call so the
// re-open cost (previously a serial per-file open_gpu_file loop) also
// parallelizes across nvme_storage's worker threads.  Files already
// open in this process are returned directly (no nvme call).
std::vector<GpuFile*> HostFsBackedBlockStorage::open_gpu_files_batch_existing_(
    const GpuFileSpec* specs, uint32_t count)
{
    std::vector<GpuFile*> results(count, nullptr);

    // Phase A: resolve each name against the log + in-process files_
    // map under mtx_ (cheap, no FS work) -- classify into "already
    // open" (fast path) vs "needs shard open" (goes to nvme batch).
    struct Pending {
        uint32_t                     result_idx;
        const PersistentGpuFileLog::Entry* entry;
    };
    std::vector<Pending> pending;
    pending.reserve(count);
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (states_.empty()) return results;
        for (uint32_t i = 0; i < count; ++i) {
            const std::string nm{specs[i].name};
            // Resolve name -> log Entry (O(1) via the log's
            // name_to_index_ map), then use the Entry's file_id to check
            // if it is already open in-process (O(1) via files_, which is
            // keyed by GpuFileId).  The previous version scanned the whole
            // files_ map by name for EVERY spec -- O(count * files_open),
            // i.e. O(N^2) when reusing a large pre-existing set inside one
            // process (an 8M-file re-open chunked into batches grows
            // files_ as it goes).  That was the multi-minute single-core
            // stall observed on the GPU_FILE_OPEN_EXISTING reuse path.
            const auto* e = states_.front()->log->find_by_name(nm);
            if (e == nullptr) continue;   // not found -- leaves results[i] == nullptr
            auto it = files_.find(e->file_id);
            if (it != files_.end()) { results[i] = it->second.get(); continue; }
            pending.push_back({i, e});
        }
    }
    if (pending.empty()) return results;

    // Phase B: flatten every pending file's shards into ONE nvme
    // open_files_batch(EXISTING) call (threaded inside nvme_storage).
    // size_bytes is unused by open_file's EXISTING branch but
    // CreateSpec requires a value.
    std::vector<INvmeStorage::CreateSpec> shard_specs;
    std::vector<uint32_t> shard_base(pending.size(), 0);
    shard_specs.reserve(pending.size() * kGpuFileMaxShards);
    for (std::size_t p = 0; p < pending.size(); ++p) {
        const auto* e = pending[p].entry;
        shard_base[p] = (uint32_t)shard_specs.size();
        for (uint32_t s = 0; s < e->num_shards; ++s) {
            auto* state = find_state(e->shard_device_ids[s]);
            if (state == nullptr) {
                shard_specs.push_back({nullptr, "", 0});   // forces failure below
                continue;
            }
            shard_specs.push_back({state->device, shard_name_(e->name, s), 0});
        }
    }
    std::vector<NvmeFile*> shard_out(shard_specs.size(), nullptr);
    (void)storage_->open_files_batch(shard_specs.data(),
                                     (uint32_t)shard_specs.size(),
                                     NVME_OPEN_EXISTING, shard_out.data());

    // Phase C: serial bookkeeping under mtx_ (fast: no FS work here).
    std::lock_guard<std::mutex> lock(mtx_);
    for (std::size_t p = 0; p < pending.size(); ++p) {
        const auto* e = pending[p].entry;
        const uint32_t ri = pending[p].result_idx;
        std::vector<NvmeFile*> shards(e->num_shards, nullptr);
        bool ok = true;
        for (uint32_t s = 0; s < e->num_shards; ++s) {
            NvmeFile* nf = shard_out[shard_base[p] + s];
            if (nf == nullptr) { ok = false; break; }
            shards[s] = nf;
        }
        if (!ok) { results[ri] = nullptr; continue; }

        auto gf = std::make_unique<GpuFile>();
        gf->id              = e->file_id;
        gf->name            = e->name;
        gf->total_size      = e->total_size;
        gf->tensor_shape[0] = e->tensor_shape[0];
        gf->tensor_shape[1] = e->tensor_shape[1];
        gf->tensor_shape[2] = e->tensor_shape[2];
        gf->shards          = std::move(shards);
        GpuFile* raw = gf.get();
        files_[e->file_id] = std::move(gf);
        results[ri] = raw;
    }
    return results;
}

bool HostFsBackedBlockStorage::close_gpu_file(GpuFile* file) {
    if (file == nullptr) return false;
    std::lock_guard<std::mutex> lock(mtx_);
    if (!is_open_) return false;
    auto it = files_.find(file->id);
    if (it == files_.end()) return false;
    bool ok = true;
    for (auto* nf : it->second->shards) {
        if (nf != nullptr && !storage_->close_file(nf)) ok = false;
    }
    files_.erase(it);
    return ok;
}

bool HostFsBackedBlockStorage::delete_gpu_file(GpuFile* file,
                                                bool persist_now) {
    if (file == nullptr) return false;
    std::lock_guard<std::mutex> lock(mtx_);
    if (!is_open_) return false;
    auto it = files_.find(file->id);
    if (it == files_.end()) return false;

    const uint32_t fid = file->id;

    // Drop this GpuFile's ShardPtrSlot residency (if any)
    // BEFORE its shards are deleted below -- otherwise it would sit
    // on a GPU slot holding pointers to shards that no longer exist
    // until the LRU eventually (but not necessarily promptly) evicts
    // it under pressure.  Default stream: delete_gpu_file has no
    // stream parameter, same convention as nvme_storage's
    // delete_file/erase_from_metadata_cache_.
    release_shard_slot_(fid, /*stream=*/nullptr);

    bool ok = true;
    for (auto* nf : it->second->shards) {
        if (nf != nullptr) {
            if (!storage_->delete_file(nf, /*persist_now=*/persist_now)) {
                ok = false;
            }
        }
    }

    for (auto& s : states_) {
        (void)s->log->remove(fid);
        s->dirty = true;
    }
    files_.erase(it);

    if (persist_now) {
        for (auto& s : states_) {
            if (!s->log->persist()) {
                std::fprintf(stderr,
                    "[block_storage] delete_gpu_file: persist dev=%d "
                    "failed; staying dirty for retry\n",
                    s->device->device_id);
                ok = false;
                continue;
            }
            s->dirty = false;
        }
    }
    return ok;
}

bool HostFsBackedBlockStorage::delete_gpu_files_batch(
    GpuFile* const* files, uint32_t count, bool* out_ok)
{
    if (count == 0) return true;
    if (files == nullptr || out_ok == nullptr) return false;
    for (uint32_t i = 0; i < count; ++i) out_ok[i] = false;

    // Phase A: under mtx_, look up + validate every file, drop its
    // ShardPtrSlot residency (cheap, mirrors delete_gpu_file), and
    // flatten all shards into ONE flat list PER DEVICE (nvme_storage's
    // delete_files_batch is per-device, so shards on different
    // devices go through separate calls -- but each of those calls is
    // itself internally multi-threaded).
    std::vector<std::vector<NvmeFile*>> shards_per_dev(states_.size());
    std::vector<std::vector<uint32_t>>  shard_result_idx(states_.size());
    std::vector<uint32_t> file_ids(count, 0);
    std::vector<bool>     file_found(count, false);
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!is_open_) return false;
        for (uint32_t i = 0; i < count; ++i) {
            if (files[i] == nullptr) continue;
            auto it = files_.find(files[i]->id);
            if (it == files_.end()) continue;
            file_found[i] = true;
            file_ids[i]   = files[i]->id;
            release_shard_slot_(file_ids[i], /*stream=*/nullptr);
            for (auto* nf : it->second->shards) {
                if (nf == nullptr) continue;
                for (std::size_t d = 0; d < states_.size(); ++d) {
                    if (states_[d]->device == nf->device) {
                        shards_per_dev[d].push_back(nf);
                        shard_result_idx[d].push_back(i);
                        break;
                    }
                }
            }
        }
    }

    // Phase B (no mtx_ held): one delete_files_batch call per device
    // (each internally multi-threaded across its own shards).  A
    // shard failure marks its owning file as failed.
    std::vector<bool> file_ok(count, true);
    for (uint32_t i = 0; i < count; ++i) if (!file_found[i]) file_ok[i] = false;
    for (std::size_t d = 0; d < states_.size(); ++d) {
        if (shards_per_dev[d].empty()) continue;
        // bool's std::vector specialization has no .data() -- use a
        // heap array instead.
        auto shard_ok = std::make_unique<bool[]>(shards_per_dev[d].size());
        (void)storage_->delete_files_batch(shards_per_dev[d].data(),
                                           (uint32_t)shards_per_dev[d].size(),
                                           shard_ok.get());
        for (std::size_t s = 0; s < shards_per_dev[d].size(); ++s) {
            if (!shard_ok[s]) file_ok[shard_result_idx[d][s]] = false;
        }
    }

    // Phase C: serial GpuFile-log bookkeeping under mtx_.  Always
    // defers persist (bulk mode) -- caller MUST flush_metadata().
    {
        std::lock_guard<std::mutex> lock(mtx_);
        for (uint32_t i = 0; i < count; ++i) {
            if (!file_found[i]) continue;
            for (auto& s : states_) {
                (void)s->log->remove(file_ids[i]);
                s->dirty = true;
            }
            files_.erase(file_ids[i]);
            out_ok[i] = file_ok[i];
        }
    }
    bool all_ok = true;
    for (uint32_t i = 0; i < count; ++i) if (!out_ok[i]) all_ok = false;
    return all_ok;
}

bool HostFsBackedBlockStorage::flush_metadata() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!is_open_) return false;
    bool ok = true;
    // Also drain underlying nvme_storage's deferred state.
    for (auto& s : states_) {
        if (!storage_->flush_metadata(s->device)) ok = false;
        if (s->dirty) {
            if (!s->log->persist()) {
                ok = false;
                continue;
            }
            s->dirty = false;
        }
    }
    return ok;
}

std::vector<std::string>
HostFsBackedBlockStorage::list_gpu_file_names() const {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<std::string> out;
    if (!is_open_ || states_.empty()) return out;
    const auto& es = states_.front()->log->entries();
    out.reserve(es.size());
    for (const auto& e : es) out.push_back(e.name);
    return out;
}

std::vector<GpuFile*>
HostFsBackedBlockStorage::list_open_gpu_files() const {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<GpuFile*> out;
    out.reserve(files_.size());
    for (const auto& [fid, gf] : files_) out.push_back(gf.get());
    return out;
}

// ---------------------------------------------------------------------------
// GPU acquire/release: pool configuration
// ---------------------------------------------------------------------------
//
// Sizes this layer's own ShardPtrSlot pool (l1_capacity GpuFiles) and
// forwards to the underlying INvmeStorage's per-shard handle-cache
// configuration.  MUST be called once, right after bootstrap(), before
// the first acquire_device_handle.
bool HostFsBackedBlockStorage::configure_handle_pool(uint32_t l1_capacity, uint32_t l2_capacity) {
    {
        std::lock_guard<std::mutex> lock(shard_pools_mtx_);
        if (!shard_pools_.empty()) {
            std::fprintf(stderr,
                "[block_storage] configure_handle_pool: called after at "
                "least one pool was already lazily initialised -- must be "
                "called right after bootstrap(), before any acquire\n");
            return false;
        }
        if (l1_capacity == 0) {
            std::fprintf(stderr,
                "[block_storage] configure_handle_pool: l1_capacity == 0\n");
            return false;
        }
        handle_pool_capacity_ = l1_capacity;
    }
    // Forward to the underlying INvmeStorage -- its
    // NvmeFileDeviceHandle cache is two-tier but counts
    // individual SHARD NvmeFiles, not GpuFiles; this layer's own
    // ShardPtrSlot pool (single-tier, l1_capacity only -- see the
    // ShardPool doc comment in the header for why a second tier
    // isn't worth it for this tiny, cheap-to-rebuild payload) counts
    // GpuFiles.  Scale by kGpuFileMaxShards (the conservative worst
    // case: every GpuFile uses the max shard count) so a batch of up
    // to l1_capacity GpuFiles can always have every shard resident
    // at once -- see block_storage.h's configure_handle_pool doc.
    if (storage_ == nullptr) {
        std::fprintf(stderr,
            "[block_storage] configure_handle_pool: not bootstrapped\n");
        return false;
    }
    return storage_->configure_handle_pool(l1_capacity * kGpuFileMaxShards,
                                           l2_capacity * kGpuFileMaxShards);
}

HostFsBackedBlockStorage::ShardPool* HostFsBackedBlockStorage::get_or_init_pool_() {
    int cuda_device = -1;
    cudaError_t cerr = cudaGetDevice(&cuda_device);
    if (cerr != cudaSuccess) {
        std::fprintf(stderr,
            "[block_storage] get_or_init_pool_: cudaGetDevice: %s\n",
            cudaGetErrorString(cerr));
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(shard_pools_mtx_);
    auto it = shard_pools_.find(cuda_device);
    if (it != shard_pools_.end()) return it->second.get();

    if (handle_pool_capacity_ == 0) {
        std::fprintf(stderr,
            "[block_storage] get_or_init_pool_: configure_handle_pool was "
            "never called (handle_pool_capacity_ == 0)\n");
        return nullptr;
    }

    auto sp = std::make_unique<ShardPool>();
    sp->gpu = std::make_unique<GpuSlotPool<ShardPtrSlot>>();
    if (!sp->gpu->init(handle_pool_capacity_, cuda_device)) {
        std::fprintf(stderr,
            "[block_storage] get_or_init_pool_: GpuSlotPool::init(cap=%u, "
            "cuda_device=%d) failed\n", handle_pool_capacity_, cuda_device);
        return nullptr;
    }
    cerr = cudaMallocHost((void**)&sp->pinned_staging,
                          (std::size_t)handle_pool_capacity_ * sizeof(ShardPtrSlot));
    if (cerr != cudaSuccess) {
        std::fprintf(stderr,
            "[block_storage] get_or_init_pool_: cudaMallocHost(%zu B) "
            "failed: %s\n",
            (std::size_t)handle_pool_capacity_ * sizeof(ShardPtrSlot),
            cudaGetErrorString(cerr));
        return nullptr;
    }
    sp->capacity = handle_pool_capacity_;
    auto* raw = sp.get();
    shard_pools_.emplace(cuda_device, std::move(sp));
    return raw;
}

ShardPtrSlot* HostFsBackedBlockStorage::resolve_shard_slot_(
    GpuFileId id, const ShardPtrSlot& value, cudaStream_t stream,
    const std::unordered_set<GpuFileId>* protect) {
    ShardPool* sp = get_or_init_pool_();
    if (sp == nullptr) return nullptr;

    std::lock_guard<std::mutex> lock(resident_mtx_);

    ShardPtrSlot* slot = nullptr;
    auto rit = resident_slot_.find(id);
    if (rit != resident_slot_.end()) {
        // Already resident -- reuse the same GPU slot, but ALWAYS
        // rewrite its content (see the ShardPool doc comment: a
        // shard's underlying pointer may have moved since last time).
        slot = rit->second;
        const uint32_t idx = (uint32_t)(slot - sp->gpu->slot_ptr(0));
        sp->pinned_staging[idx] = value;
        cudaError_t cerr = cudaMemcpyAsync(slot, &sp->pinned_staging[idx],
                                          sizeof(ShardPtrSlot),
                                          cudaMemcpyHostToDevice, stream);
        if (cerr != cudaSuccess) {
            std::fprintf(stderr,
                "[block_storage] resolve_shard_slot_: cudaMemcpyAsync "
                "(refresh) failed: %s\n", cudaGetErrorString(cerr));
            return nullptr;
        }
        sp->gpu->mark_filled(idx, stream);
        // touch LRU
        auto lit = lru_pos_.find(id);
        if (lit != lru_pos_.end()) lru_.erase(lit->second);
        lru_.push_front(id);
        lru_pos_[id] = lru_.begin();
        return slot;
    }

    // Not resident: reserve a slot (evicting the LRU tail if the
    // pool is full, skipping any id in `protect`), write our pinned
    // staging, queue the copy.
    uint32_t idx;
    slot = sp->gpu->reserve_slot(stream, &idx);
    if (slot == nullptr) {
        // Scan from the LRU tail (oldest) for the first entry NOT in
        // `protect` -- mirrors TieredHandleCache::evict_l1_lru_one_locked_.
        GpuFileId victim = 0;
        bool found = false;
        for (auto it = lru_.rbegin(); it != lru_.rend(); ++it) {
            if (protect != nullptr && protect->count(*it) != 0) continue;
            victim = *it;
            found = true;
            break;
        }
        if (!found) return nullptr;   // pool exhausted, nothing evictable
        lru_pos_.erase(victim);
        lru_.remove(victim);
        auto vit = resident_slot_.find(victim);
        if (vit != resident_slot_.end()) {
            sp->gpu->release_async(vit->second, stream);
            resident_slot_.erase(vit);
        }
        slot = sp->gpu->reserve_slot(stream, &idx);
        if (slot == nullptr) return nullptr;
    }
    sp->pinned_staging[idx] = value;
    cudaError_t cerr = cudaMemcpyAsync(slot, &sp->pinned_staging[idx],
                                      sizeof(ShardPtrSlot),
                                      cudaMemcpyHostToDevice, stream);
    if (cerr != cudaSuccess) {
        std::fprintf(stderr,
            "[block_storage] resolve_shard_slot_: cudaMemcpyAsync failed: "
            "%s\n", cudaGetErrorString(cerr));
        sp->gpu->release_async(slot, stream);
        return nullptr;
    }
    sp->gpu->mark_filled(idx, stream);

    resident_slot_[id] = slot;
    lru_.push_front(id);
    lru_pos_[id] = lru_.begin();
    return slot;
}

void HostFsBackedBlockStorage::release_shard_slot_(GpuFileId id, cudaStream_t stream) {
    ShardPool* sp = get_or_init_pool_();
    if (sp == nullptr) return;

    std::lock_guard<std::mutex> lock(resident_mtx_);
    auto rit = resident_slot_.find(id);
    if (rit == resident_slot_.end()) return;   // not resident -- nothing to do
    sp->gpu->release_async(rit->second, stream);
    resident_slot_.erase(rit);
    auto lit = lru_pos_.find(id);
    if (lit != lru_pos_.end()) { lru_.erase(lit->second); lru_pos_.erase(lit); }
}

// ---------------------------------------------------------------------------
// GPU acquire/release (two-tier, async, pooled)
// ---------------------------------------------------------------------------
//
// Per-shard fan-out over the underlying INvmeStorage's handle-cache API
// (ensure-resident, near-free on a hit).  This layer's own
// ShardPtrSlot is single-tier (see the ShardPool doc comment in the
// header) and ALWAYS rewritten on every acquire -- cheap (~32 B copy)
// and avoids caching a shard pointer that may have moved underneath
// nvme_storage's own cache.
//
// io_engine reads d_shards_host[shard_idx] on the host when
// staging per-IO GPUIoContext arrays, copying the resolved
// NvmeFileDeviceHandle* directly into each ctx before cudaMemcpy.
// The kernel never indirects through a GPU-resident shard table:
// one less GPU load per IO and no shard-table lifetime to manage.
//
// Failure semantics:
//   - If any per-shard acquire fails, all previously acquired shard
//     handles are released (advisory) and the function returns
//     nullptr.
//   - The returned GpuFileHandle is heap-allocated; release_device_handle
//     is the matching deleter (also an advisory release for every
//     shard handle) -- does NOT touch file->shards or files_.
//   - Idempotent against concurrent close_gpu_file: once a caller
//     has a GpuFile* and calls acquire on it, the underlying
//     NvmeFile shard pointers stay valid for the lifetime of the
//     returned GpuFileHandle (caller must release before close).
GpuFileHandle*
HostFsBackedBlockStorage::acquire_device_handle(GpuFile* file, GpuStreamHandle stream_h) {
    if (file == nullptr) {
        std::fprintf(stderr,
            "[block_storage] acquire_device_handle: file == nullptr\n");
        return nullptr;
    }
    if (storage_ == nullptr) {
        std::fprintf(stderr,
            "[block_storage] acquire_device_handle: not bootstrapped\n");
        return nullptr;
    }

    const uint32_t num_shards = file->tensor_shape[0];
    if (num_shards == 0 || file->shards.size() != num_shards) {
        std::fprintf(stderr,
            "[block_storage] acquire_device_handle: file '%s' has "
            "tensor_shape[0]=%u but %zu shards\n",
            file->name.c_str(), num_shards, file->shards.size());
        return nullptr;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_h);

    auto handle = std::make_unique<GpuFileHandle>();
    handle->file        = file;
    handle->tensor_size = file->tensor_shape[2];
    handle->num_shards  = num_shards;
    handle->d_shards_host.reserve(num_shards);
    handle->d_shards_dev = nullptr;

    // Per-shard "ensure resident" (idempotent -- near-free on a
    // cache hit inside nvme_storage's own two-tier cache).  On
    // failure, advisory-release every shard handle already ensured
    // before returning nullptr.
    for (uint32_t s = 0; s < num_shards; ++s) {
        NvmeFile* shard = file->shards[s];
        if (shard == nullptr) {
            std::fprintf(stderr,
                "[block_storage] acquire_device_handle: file '%s' "
                "shard %u is null\n",
                file->name.c_str(), s);
            for (uint32_t p = 0; p < s; ++p) storage_->release_device_handle(file->shards[p], stream_h);
            return nullptr;
        }
        NvmeFileDeviceHandle* dh = storage_->acquire_device_handle(shard, stream_h);
        if (dh == nullptr) {
            std::fprintf(stderr,
                "[block_storage] acquire_device_handle: "
                "nvme_storage->acquire_device_handle failed for "
                "file='%s' shard=%u name='%s'\n",
                file->name.c_str(), s, shard->name.c_str());
            for (uint32_t p = 0; p < s; ++p) storage_->release_device_handle(file->shards[p], stream_h);
            return nullptr;
        }
        handle->d_shards_host.push_back(dh);
    }

    // Resolve/refresh this GpuFile's own ShardPtrSlot.
    ShardPtrSlot slot_value{};   // zero-inits every ptrs[i], including the
                                 // kGpuFileMaxShards - num_shards trailing
                                 // slots the kernel never reads.
    for (uint32_t s = 0; s < num_shards; ++s) slot_value.ptrs[s] = handle->d_shards_host[s];

    ShardPtrSlot* d_slot = resolve_shard_slot_(file->id, slot_value, stream);
    if (d_slot == nullptr) {
        std::fprintf(stderr,
            "[block_storage] acquire_device_handle: shard-pointer-table "
            "pool exhausted (capacity=%u) -- caller should evict a cached "
            "handle and retry\n", handle_pool_capacity_);
        for (uint32_t s = 0; s < num_shards; ++s)
            storage_->release_device_handle(file->shards[s], stream_h);
        return nullptr;
    }
    handle->d_shards_dev = reinterpret_cast<NvmeFileDeviceHandle**>(d_slot);

    return handle.release();
}

void HostFsBackedBlockStorage::release_device_handle(GpuFileHandle* h, GpuStreamHandle stream_h) {
    if (h == nullptr) return;
    if (storage_ == nullptr) {
        // Storage already shut down.  We must not touch GPU memory
        // because the pools it and this layer own are invalid after
        // shutdown.  Best we can do is leak the GpuFileHandle struct
        // itself.
        std::fprintf(stderr,
            "[block_storage] release_device_handle called after "
            "shutdown; leaking GpuFileHandle to avoid use-after-free\n");
        return;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_h);

    // Advisory -- lets this GpuFile's ShardPtrSlot (and every
    // shard's NvmeFileDeviceHandle) be evicted sooner.  Nothing is
    // freed synchronously.
    if (h->file != nullptr) release_shard_slot_(h->file->id, stream);
    h->d_shards_dev = nullptr;
    for (uint32_t s = 0; s < h->num_shards && h->file != nullptr && s < h->file->shards.size(); ++s) {
        storage_->release_device_handle(h->file->shards[s], stream_h);
    }
    delete h;
}

// ---------------------------------------------------------------------------
// GPU acquire (batch)
// ---------------------------------------------------------------------------
//
// Flattens every requested file's shards into ONE list and issues a
// SINGLE INvmeStorage::acquire_device_handles_batch call, instead of
// `count` separate acquire_device_handle fan-outs -- this is what
// actually avoids "one IO batch launches thousands of small cache
// lookups/copies" at the layer that does the expensive work (FIEMAP
// walk on a COLD miss).  This layer's own ShardPtrSlot resolution
// stays per-file (each is a cheap ~32 B copy -- batching THAT isn't
// worth the complexity), but must protect every file in this batch
// from self-eviction while resolving them one at a time (mirrors
// TieredHandleCache's batch protect-set fix).
bool HostFsBackedBlockStorage::acquire_device_handles_batch(
    GpuFile* const* files, uint32_t count, GpuStreamHandle stream_h,
    GpuFileHandle** out_handles)
{
    if (count == 0) return true;
    if (files == nullptr || out_handles == nullptr) {
        std::fprintf(stderr,
            "[block_storage] acquire_device_handles_batch: null files/out_handles\n");
        return false;
    }
    if (storage_ == nullptr) {
        std::fprintf(stderr,
            "[block_storage] acquire_device_handles_batch: not bootstrapped\n");
        return false;
    }
    if (count > handle_pool_capacity_) {
        std::fprintf(stderr,
            "[block_storage] acquire_device_handles_batch: count=%u > "
            "l1_capacity=%u -- a single batch's distinct GpuFiles must fit "
            "in this layer's ShardPtrSlot pool for the duration of the call "
            "(raise l1_capacity)\n", count, handle_pool_capacity_);
        return false;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_h);

    // 1. Flatten every file's shards into one list for a single
    //    nvme_storage batch acquire.
    std::vector<NvmeFile*> flat_shards;
    flat_shards.reserve((std::size_t)count * kGpuFileMaxShards);
    for (uint32_t i = 0; i < count; ++i) {
        GpuFile* f = files[i];
        if (f == nullptr) {
            std::fprintf(stderr,
                "[block_storage] acquire_device_handles_batch: files[%u] == "
                "nullptr\n", i);
            return false;
        }
        const uint32_t ns = f->tensor_shape[0];
        if (ns == 0 || f->shards.size() != ns) {
            std::fprintf(stderr,
                "[block_storage] acquire_device_handles_batch: file '%s' "
                "has tensor_shape[0]=%u but %zu shards\n",
                f->name.c_str(), ns, f->shards.size());
            return false;
        }
        for (uint32_t s = 0; s < ns; ++s) {
            NvmeFile* shard = f->shards[s];
            if (shard == nullptr) {
                std::fprintf(stderr,
                    "[block_storage] acquire_device_handles_batch: file "
                    "'%s' shard %u is null\n", f->name.c_str(), s);
                return false;
            }
            flat_shards.push_back(shard);
        }
    }

    std::vector<NvmeFileDeviceHandle*> flat_out(flat_shards.size(), nullptr);
    if (!flat_shards.empty() &&
        !storage_->acquire_device_handles_batch(
            flat_shards.data(), (uint32_t)flat_shards.size(), stream_h,
            flat_out.data())) {
        std::fprintf(stderr,
            "[block_storage] acquire_device_handles_batch: nvme_storage "
            "batch acquire failed (%zu flattened shards)\n",
            flat_shards.size());
        return false;
    }

    // 2. Build one GpuFileHandle per file, scattering the flat
    //    results back, then resolve each file's ShardPtrSlot --
    //    protecting every id in this batch from self-eviction.
    std::unordered_set<GpuFileId> protect;
    protect.reserve(count);
    for (uint32_t i = 0; i < count; ++i) protect.insert(files[i]->id);

    std::size_t flat_idx = 0;
    for (uint32_t i = 0; i < count; ++i) {
        GpuFile* f = files[i];
        const uint32_t ns = f->tensor_shape[0];

        auto handle = std::make_unique<GpuFileHandle>();
        handle->file        = f;
        handle->tensor_size = f->tensor_shape[2];
        handle->num_shards  = ns;
        handle->d_shards_host.reserve(ns);
        handle->d_shards_dev = nullptr;

        ShardPtrSlot slot_value{};
        for (uint32_t s = 0; s < ns; ++s) {
            NvmeFileDeviceHandle* dh = flat_out[flat_idx++];
            handle->d_shards_host.push_back(dh);
            slot_value.ptrs[s] = dh;
        }

        ShardPtrSlot* d_slot = resolve_shard_slot_(f->id, slot_value, stream, &protect);
        if (d_slot == nullptr) {
            std::fprintf(stderr,
                "[block_storage] acquire_device_handles_batch: "
                "shard-pointer-table pool exhausted for file '%s' "
                "(capacity=%u)\n", f->name.c_str(), handle_pool_capacity_);
            // 隐患-3 fix (R12 A-4): clean up handles already acquired
            // for files [0..i) so the caller doesn't leak host
            // GpuFileHandle structs when treating false as "whole
            // batch failed".
            for (uint32_t j = 0; j < i; ++j) {
                release_device_handle(out_handles[j], stream_h);
                out_handles[j] = nullptr;
            }
            return false;
        }
        handle->d_shards_dev = reinterpret_cast<NvmeFileDeviceHandle**>(d_slot);
        out_handles[i] = handle.release();
    }
    return true;
}

}  // namespace tutti
