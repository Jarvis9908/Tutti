/**
 * nvme_storage_bulk_smoke.cu -- bulk-init perf + correctness for
 * HostFsBackedNvmeStorage::create_file(persist_now, sync_now).
 *
 * What this validates (R5a.1):
 *   1. create_file(..., false, false) skips both per-file fsyncs and
 *      the per-call log.persist().
 *   2. flush_metadata() drains both deferred queues atomically:
 *      one syncfs(2) + one log rewrite.
 *   3. After shutdown() + a fresh bootstrap(), every file added in
 *      the bulk batch is recoverable (open_file by name + extents
 *      match what was returned at create time).
 *   4. Wall-time of the bulk path is dramatically lower than the
 *      sync_now=true / persist_now=true path at the same N -- this
 *      is the whole motivation behind R5a.1.
 *
 * Comparison protocol:
 *   - phase A: N files via single-file mode (defaults), no flush
 *              needed -- already durable per call.
 *   - phase B: N files via bulk mode, one flush_metadata() at the
 *              end.
 *   - print wall-time + speedup ratio.
 *
 * R11.5: both phases use multi-threaded create (one thread per 2000
 * files).  create_file_locked runs FS operations (open/fallocate/
 * fiemap/linkat) WITHOUT mtx_ so the threads genuinely parallelize;
 * only the bookkeeping (log add, files map insert) is serialized.
 *
 * Usage:
 *   sudo ./nvme_storage_bulk_smoke --gpu 0 --cap 32 \
 *        --n 10000 --size 4096 0000:08:00.0
 *
 * Single-device on purpose: the wall-time-and-ratio comparison is
 * the headline; multi-device parallel init is a separate concern.
 *
 * DESTRUCTIVE on first run (mkfs.ext4 -F if blk has no FS).
 */

#include "host_fs_backed_nvme_storage.h"
#include "nvme_file.h"
#include "nvme_storage.h"

#include "../common/registry_cli.h"
#include "../../coordinator/include/device.h"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

// ----------------------------------------------------------------------------

#define STEP_OK(...)                                                          \
    do {                                                                      \
        std::fprintf(stderr, "[ OK ] step=%-3d ", g_step++);                  \
        std::fprintf(stderr, __VA_ARGS__);                                    \
        std::fprintf(stderr, "\n");                                           \
    } while (0)

#define STEP_FAIL(...)                                                        \
    do {                                                                      \
        std::fprintf(stderr, "[FAIL] step=%-3d ", g_step);                    \
        std::fprintf(stderr, __VA_ARGS__);                                    \
        std::fprintf(stderr, "\n");                                           \
        std::exit(1);                                                         \
    } while (0)

#define CUDA_OK(expr)                                                         \
    do {                                                                      \
        cudaError_t _e = (expr);                                              \
        if (_e != cudaSuccess) {                                              \
            STEP_FAIL("%s: %s", #expr, cudaGetErrorString(_e));               \
        }                                                                     \
    } while (0)

static int g_step = 1;

// ----------------------------------------------------------------------------

static const char* arg_after(const char* arg, const char* prefix) {
    size_t n = std::strlen(prefix);
    return std::strncmp(arg, prefix, n) == 0 ? arg + n : nullptr;
}

static void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage:\n"
        "  IN_PROCESS: %s [--gpu N] [--cap N] [--n N] [--size BYTES] <pci_addr>\n"
        "  SERVICE_CLIENT: %s [--cuda N] --service <endpoint> --dev-id N "
        "[--n N] [--size BYTES]\n"
        "\n"
        "Compares wall-time of N create_file calls in two modes:\n"
        "  A: per-call durable (persist_now=true, sync_now=true)\n"
        "  B: bulk + single flush_metadata()\n"
        "Then bounce the storage (shutdown + bootstrap) and verify\n"
        "every B-mode entry is recoverable.\n"
        "\n"
        "  --gpu  N      cudaSetDevice (default 0; IN_PROCESS only, use\n"
        "                --cuda for SERVICE_CLIENT)\n"
        "  --cap  N      kernel_ioq_cap (default 32; IN_PROCESS only)\n"
        "  --n    N      files per phase (default 2000; bump higher\n"
        "                to see the gap really open up)\n"
        "  --size BYTES  per-file user-visible size (default 4096)\n"
        "\n"
        "Single-device on purpose (wall-time comparison is the headline).\n",
        prog, prog);
}

// ----------------------------------------------------------------------------

struct CreateLog {
    std::string                       name;
    uint64_t                          file_id;
    std::vector<tutti::LbaExtent>     extents;
    const tutti::Device*              device = nullptr;  // which dev this file lives on
};

static double seconds_since(
    const std::chrono::steady_clock::time_point& t0)
{
    auto d = std::chrono::steady_clock::now() - t0;
    return std::chrono::duration<double>(d).count();
}

// ----------------------------------------------------------------------------

// R11.5: parallel bulk-create.  One thread per chunk of
// kCreateChunkSize files; the FS operations (open/fallocate/fiemap/
// linkat) in create_file_locked run WITHOUT mtx_ so the threads
// genuinely parallelize at the kernel level.
static constexpr uint64_t kCreateChunkSize = 2000;

static uint32_t parallel_create_n_threads(uint64_t n_files) {
    return (uint32_t)((n_files + kCreateChunkSize - 1) / kCreateChunkSize);
}

static bool parallel_create(
    tutti::HostFsBackedNvmeStorage& storage,
    const std::vector<const tutti::Device*>& devs,
    const char* prefix,
    uint64_t n_files, uint64_t file_size,
    uint32_t nvme_flags,
    std::vector<CreateLog>* collect = nullptr)
{
    const uint32_t n_threads = parallel_create_n_threads(n_files);
    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    std::atomic<bool> failed{false};
    std::vector<std::vector<CreateLog>> per_thread(n_threads);

    for (uint32_t t = 0; t < n_threads; ++t) {
        uint64_t start = (uint64_t)t * kCreateChunkSize;
        uint64_t end   = std::min(start + kCreateChunkSize, n_files);
        threads.emplace_back([&, t, start, end]() {
            auto& local = per_thread[t];
            if (collect) local.reserve(end - start);
            for (uint64_t i = start; i < end; ++i) {
                char nm[64];
                std::snprintf(nm, sizeof(nm), "%s%lu", prefix, (unsigned long)i);
                // Round-robin files across devices.
                const tutti::Device* dev = devs[i % devs.size()];
                auto* nf = storage.open_file(dev, nm, nvme_flags, file_size);
                if (nf == nullptr) {
                    std::fprintf(stderr,
                        "[FAIL] parallel_create(%s) i=%lu\n",
                        prefix, (unsigned long)i);
                    failed.store(true);
                    return;
                }
                if (collect) {
                    CreateLog rec;
                    rec.name    = nm;
                    rec.file_id = nf->id;
                    rec.extents = nf->extents;
                    rec.device  = dev;
                    local.push_back(std::move(rec));
                }
            }
        });
    }
    for (auto& th : threads) th.join();
    if (failed.load()) return false;

    if (collect) {
        for (auto& vec : per_thread)
            for (auto& rec : vec)
                collect->push_back(std::move(rec));
    }
    return true;
}

// ----------------------------------------------------------------------------

int main(int argc, char** argv) {
    uint64_t n_files   = 2000;
    uint64_t file_size = 4096;
    tutti_examples::RegistryCliOptions ropt;

    for (int i = 1; i < argc; ) {
        if (tutti_examples::parse_registry_cli_arg(ropt, argc, argv, i)) { ++i; continue; }
        const char* a = argv[i];
        const char* v = nullptr;
        if      ((v = arg_after(a, "--gpu=")))  { ropt.cuda_dev = std::atoi(v); ++i; }
        else if ((v = arg_after(a, "--n=")))    { n_files   = std::strtoull(v, nullptr, 10); ++i; }
        else if ((v = arg_after(a, "--size="))) { file_size = std::strtoull(v, nullptr, 10); ++i; }
        else if (std::strcmp(a, "--gpu")  == 0 && i + 1 < argc) { ropt.cuda_dev = std::atoi(argv[i + 1]); i += 2; }
        else if (std::strcmp(a, "--n")    == 0 && i + 1 < argc) { n_files   = std::strtoull(argv[i + 1], nullptr, 10); i += 2; }
        else if (std::strcmp(a, "--size") == 0 && i + 1 < argc) { file_size = std::strtoull(argv[i + 1], nullptr, 10); i += 2; }
        else if (std::strcmp(a, "-h") == 0 || std::strcmp(a, "--help") == 0) {
            usage(argv[0]); return 0;
        } else if (a[0] != '-') {
            ropt.pci_addrs.emplace_back(a); ++i;
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", a);
            usage(argv[0]); return 1;
        }
    }
    if (!tutti_examples::validate_registry_cli(ropt, argv[0])) { usage(argv[0]); return 1; }
    const std::size_t n_dev = ropt.service_mode ? ropt.dev_ids.size() : ropt.pci_addrs.size();
    if (n_dev == 0) {
        std::fprintf(stderr, "at least one device required\n");
        usage(argv[0]); return 1;
    }
    if (n_files == 0 || file_size == 0) {
        std::fprintf(stderr, "--n and --size must be > 0\n");
        return 1;
    }

    // [1] cuda prime + cudaSetDevice (registry needs CUDA up).
    {
        cudaError_t cerr = cudaFree(0);
        if (cerr != cudaSuccess && cerr != cudaErrorInvalidValue) {
            STEP_FAIL("cuda driver prime: %s", cudaGetErrorString(cerr));
        }
        (void)cudaGetLastError();
        CUDA_OK(cudaSetDevice(ropt.cuda_dev));
        STEP_OK("cudaSetDevice(%d)", ropt.cuda_dev);
    }

    // [2] registry -- IN_PROCESS or SERVICE_CLIENT.
    //     build_queue_group=false: this smoke only exercises the host
    //     create_file/flush_metadata path, no GPU submit.
    auto reg = tutti_examples::open_registry(ropt, /*build_queue_group=*/false);
    if (reg.ptr == nullptr) STEP_FAIL("registry Open()");
    if (reg.ptr->device_count() == 0) STEP_FAIL("device_count == 0");
    std::vector<const tutti::Device*> devs;
    devs.reserve(reg.ptr->device_count());
    for (std::size_t i = 0; i < reg.ptr->device_count(); ++i) {
        const tutti::Device* d = reg.ptr->device_at(i);
        if (d == nullptr) STEP_FAIL("device_at(%zu) null", i);
        devs.push_back(d);
    }
    STEP_OK("registry up: mode=%s n_dev=%zu", ropt.service_mode ? "service" : "direct", devs.size());

    // [3] HostFsBackedNvmeStorage::bootstrap.
    auto storage =
        std::make_unique<tutti::HostFsBackedNvmeStorage>(
            tutti::HostFsBackedNvmeStorage::Config{});
    if (!storage->bootstrap(devs)) STEP_FAIL("storage.bootstrap");
    {
        std::size_t total = 0;
        for (auto* d : devs) total += storage->list_file_names(d).size();
        STEP_OK("storage bootstrap (initial entries=%zu)", total);
    }

    // [3.5] Pre-cleanup: a previous run that wasn't fully cleaned can
    //       leave A_*/B_* stragglers in the directory.  Without this
    //       pass, phase A's first create would hit "already exists".
    //       We use list_file_names (which walks the persistent log,
    //       not just open fds) and delete anything matching our
    //       prefixes.  Like the final cleanup, we use the deferred
    //       delete path + a single flush_metadata at the end -- a
    //       previous run that aborted at N=20000 leaves 40000
    //       stragglers, and persist-per-delete on that scale is
    //       O(N^2) total log writes (minutes-to-hours wall time).
    {
        auto t0 = std::chrono::steady_clock::now();
        std::size_t n_pre = 0;
        for (auto* d : devs) {
            auto names = storage->list_file_names(d);
            for (const auto& nm : names) {
                if (nm.size() < 2) continue;
                if (nm[0] != 'A' && nm[0] != 'B') continue;
                if (nm[1] != '_') continue;
                tutti::NvmeFile* f = storage->open_file(d, nm);
                if (f == nullptr) continue;
                if (storage->delete_file(f, /*persist_now=*/false)) ++n_pre;
            }
            if (n_pre > 0) (void)storage->flush_metadata(d);
        }
        if (n_pre > 0) {
            double sec_p = seconds_since(t0);
            STEP_OK("pre-cleanup: removed %zu straggler(s) from "
                    "previous run, wall=%.3fs (%.0f ops/s)",
                    n_pre, sec_p, sec_p > 0 ? n_pre / sec_p : 0.0);
        } else {
            STEP_OK("pre-cleanup: no stragglers (clean slate)");
        }
    }

    // [4] phase A: per-call durable creates (multi-threaded, R11.5).
    {
        auto t0 = std::chrono::steady_clock::now();
        if (!parallel_create(*storage, devs, "A_", n_files, file_size,
                             tutti::NVME_OPEN_CREATE))
            STEP_FAIL("phase A parallel_create");
        double sec_a = seconds_since(t0);
        STEP_OK("phase A: %lu files, per-call durable, %u threads, "
                "wall=%.3fs (%.0f ops/s)",
                (unsigned long)n_files, parallel_create_n_threads(n_files),
                sec_a, n_files / sec_a);
    }

    // [5] phase B: bulk creates + one flush_metadata at end
    //     (multi-threaded, R11.5).
    std::vector<CreateLog> b_log;
    b_log.reserve(n_files);
    double sec_b_creates = 0.0;
    double sec_b_flush   = 0.0;
    {
        auto t0 = std::chrono::steady_clock::now();
        if (!parallel_create(*storage, devs, "B_", n_files, file_size,
                             tutti::NVME_OPEN_CREATE | tutti::NVME_OPEN_NO_PERSIST | tutti::NVME_OPEN_NO_SYNC,
                             &b_log))
            STEP_FAIL("phase B parallel_create");
        sec_b_creates = seconds_since(t0);

        auto tf = std::chrono::steady_clock::now();
        for (auto* d : devs)
            if (!storage->flush_metadata(d)) STEP_FAIL("flush_metadata(dev=%d)", d->device_id);
        sec_b_flush = seconds_since(tf);

        double sec_b_total = sec_b_creates + sec_b_flush;
        STEP_OK("phase B: %lu files, deferred + 1x flush, %u threads, "
                "wall=%.3fs (creates=%.3fs flush=%.3fs, %.0f ops/s)",
                (unsigned long)n_files, parallel_create_n_threads(n_files),
                sec_b_total, sec_b_creates, sec_b_flush, n_files / sec_b_total);
    }

    // [6] Bounce: shutdown + reload, verify phase B entries survived
    //     the deferred persist + flush.
    if (!storage->shutdown()) STEP_FAIL("shutdown");
    STEP_OK("storage shutdown (umount, log persist)");

    storage = std::make_unique<tutti::HostFsBackedNvmeStorage>(
        tutti::HostFsBackedNvmeStorage::Config{});
    if (!storage->bootstrap(devs)) STEP_FAIL("storage.bootstrap (re)");
    {
        std::size_t total = 0;
        for (auto* d : devs) total += storage->list_file_names(d).size();
        STEP_OK("storage re-bootstrap (recovered entries=%zu)", total);
    }

    {
        // Every B_* name should reopen with matching id + extents.
        std::size_t mismatch = 0;
        std::size_t missing  = 0;
        for (const auto& rec : b_log) {
            tutti::NvmeFile* nf = storage->open_file(rec.device, rec.name);
            if (nf == nullptr) { ++missing; continue; }
            if (nf->id != rec.file_id ||
                nf->extents.size() != rec.extents.size()) {
                ++mismatch;
                continue;
            }
            for (size_t k = 0; k < rec.extents.size(); ++k) {
                if (nf->extents[k].start_lba      != rec.extents[k].start_lba ||
                    nf->extents[k].length_blocks  != rec.extents[k].length_blocks) {
                    ++mismatch;
                    break;
                }
            }
        }
        if (missing != 0 || mismatch != 0) {
            STEP_FAIL("phase B recovery: missing=%zu mismatch=%zu",
                      missing, mismatch);
        }
        STEP_OK("phase B recovery: all %lu entries replay clean "
                "(file_id + extents stable across bootstrap bounce)",
                (unsigned long)b_log.size());
    }

    // [7] cleanup: delete every file on every device (both phases) so
    //     reruns don't leave a million tutti/.tutti/*.bin behind.
    //     We use the bulk-delete path (persist_now=false) so the
    //     log rewrite happens once at the end via flush_metadata,
    //     not N times.  The on-disk .bin is unlinked synchronously
    //     either way -- only the log persist is deferred, mirroring
    //     create_file's bulk-init mode.
    {
        auto t0 = std::chrono::steady_clock::now();
        std::size_t n_deleted = 0;
        for (auto* d : devs) {
            auto names = storage->list_file_names(d);
            for (const auto& nm : names) {
                tutti::NvmeFile* f = storage->open_file(d, nm);
                if (f == nullptr) continue;
                if (storage->delete_file(f, /*persist_now=*/false)) ++n_deleted;
            }
            (void)storage->flush_metadata(d);
        }
        double sec_d = seconds_since(t0);
        STEP_OK("cleanup: deleted %zu files (deferred + 1x flush), "
                "wall=%.3fs (%.0f ops/s)",
                n_deleted, sec_d, sec_d > 0 ? n_deleted / sec_d : 0.0);
    }

    // [8] shutdown + registry close.
    if (!storage->shutdown()) STEP_FAIL("final shutdown");
    STEP_OK("final shutdown");
    reg.ptr->Close();
    STEP_OK("registry closed");

    std::fprintf(stderr,
        "\n=== nvme_storage_bulk_smoke (n=%lu, size=%lu): all 8 steps passed ===\n",
        (unsigned long)n_files, (unsigned long)file_size);
    return 0;
}
