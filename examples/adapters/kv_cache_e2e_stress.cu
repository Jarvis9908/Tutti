/**
 * kv_cache_e2e_stress.cu -- large-scale end-to-end stress test through
 * the full Coordinator + KvCacheIoAdapter stack (MLA layout, stream
 * overlap, and scale scenarios).
 *
 * Uses ONLY adapters/kv_cache + Coordinator public interfaces (no
 * direct nvme_storage/block_storage access, except mount_path/
 * write_blocking/read_blocking for the crash-consistency phase which
 * has no adapter-level equivalent).
 *
 * Default config: 8M GpuFiles, shape 2×8×128KB (2 shards K/V, 8 layers,
 * 128KB per tensor = 2MB/file), --data-gb 0 (== cover ALL files, so a
 * full run touches ~16TB write + ~16TB read), 2 CUDA streams.
 *
 * Phases:
 *   1. Batch create N GpuFiles via Coordinator::open_gpu_files_batch
 *      (chunked into --chunk-size groups for progress visibility; a
 *      matching-count re-run reuses the existing files instead of
 *      re-creating them, so a --batch/--l1-mib/--l2-mib sweep across
 *      multiple invocations only pays the create cost once).
 *   2. Multi-stream, multi-layer GPU R/W via KvCacheIoAdapter:
 *      - n_layers layers per file, each with K+V (or unified for MLA)
 *      - num_streams CUDA streams alternating across batches (tests
 *        cross-stream handle-cache correctness)
 *      - --batch is the concurrency knob: number of GpuFiles in flight
 *        per R/W round (registered tensors + one nvme_storage batch
 *        call per layer)
 *      - --data-gb caps total transfer; 0 (default) means cover every
 *        created file, so metadata scale == data scale
 *      - access order is shuffled by default (--seed / --no-shuffle)
 *        so the SSD sees a non-sequential GpuFileId pattern
 *      - Per-layer, per-K/V seed verification (catches cross-contamination)
 *   3. Crash consistency: ghost sweep, external rm, data bounce.
 *      Skippable with --skip-crash-consistency (e.g. for sweep runs
 *      that reuse the file set across --batch/cache-size variants).
 *   4. Bulk delete all GpuFiles.  Skippable with --skip-delete to
 *      leave the file set in place for the next sweep invocation.
 *
 * Usage (SERVICE_CLIENT):
 *   sudo ./bin/kv_cache_e2e_stress --cuda 0 \
 *       --service 127.0.0.1:50051 --dev-id 0 --dev-id 1 \
 *       [--n 8000000] [--batch 500] [--layers 8] [--tensor-kb 128] \
 *       [--data-gb 0] [--streams 2] [--l1-mib 512] [--l2-mib 2048] [--mla] \
 *       [--seed 42] [--no-shuffle] [--chunk-size 500000] \
 *       [--skip-crash-consistency] [--skip-delete] [--log <path>]
 *
 * Log output: written to <repo_root>/kv_cache_e2e_stress.log by
 * default, or to --log <path> (in addition to stderr).  Progress
 * lines ("Phase N progress: ...") are emitted at least every ~30s so
 * a long (multi-hour, TB-scale) run's liveness can be checked with
 * `tail -f` without waiting for phase completion.
 */

#include "../../coordinator/include/coordinator.h"
#include "../../coordinator/include/coordinator_config.h"
#include "../../coordinator/include/device.h"
#include "../../block_storage/include/block_storage.h"
#include "../../block_storage/include/host_fs_backed_block_storage.h"
#include "../../nvme_storage/include/nvme_file.h"
#include "../../nvme_storage/include/host_fs_backed_nvme_storage.h"
#include "../../memory/include/memory_subsystem.h"
#include "../../memory/include/memory_region.h"
#include "kv_cache_io_adapter.h"

#include "../common/registry_cli.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Dual stderr + logfile output.
// ---------------------------------------------------------------------------

static FILE* g_logfile = nullptr;
static int   g_step    = 0;

static void log_line(const char* prefix, const char* buf) {
    std::fprintf(stderr, "%s%s\n", prefix, buf);
    if (g_logfile) {
        std::fprintf(g_logfile, "%s%s\n", prefix, buf);
        std::fflush(g_logfile);
    }
}

#define STEP_OK(...) do { \
    char _buf[1024]; std::snprintf(_buf, sizeof(_buf), __VA_ARGS__); \
    char _pfx[32]; std::snprintf(_pfx, sizeof(_pfx), "[ OK ] step=%-3d ", ++g_step); \
    log_line(_pfx, _buf); \
} while (0)

#define STEP_FAIL(...) do { \
    char _buf[1024]; std::snprintf(_buf, sizeof(_buf), __VA_ARGS__); \
    char _pfx[32]; std::snprintf(_pfx, sizeof(_pfx), "[FAIL] step=%-3d ", ++g_step); \
    log_line(_pfx, _buf); \
    std::_Exit(2); \
} while (0)

#define LOG_INFO(...) do { \
    char _buf[1024]; std::snprintf(_buf, sizeof(_buf), __VA_ARGS__); \
    log_line("[INFO] ", _buf); \
} while (0)

#define CUDA_OK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) STEP_FAIL("CUDA: %s (%s)", #call, cudaGetErrorString(_e)); \
} while (0)

static double seconds_since(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
}

// Minimum interval between "still alive" progress log lines within a
// long-running phase (Phase 1 create, Phase 2 R/W, Phase 4 delete),
// independent of any batch/chunk-count based cadence -- guarantees a
// liveness signal even if individual batches/chunks take much longer
// or shorter than expected.
static constexpr double kProgressHeartbeatSec = 30.0;

static std::string format_duration(double secs) {
    if (secs < 0) secs = 0;
    uint64_t s = (uint64_t)secs;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%02lu:%02lu:%02lu",
                  (unsigned long)(s / 3600), (unsigned long)((s % 3600) / 60),
                  (unsigned long)(s % 60));
    return std::string(buf);
}

// O_DIRECT (used by nvme_storage's read_blocking/write_blocking) requires
// a block-aligned buffer.
struct AlignedBuf {
    uint8_t* p = nullptr;
    size_t   n = 0;
    explicit AlignedBuf(size_t bytes, size_t align = 4096) : n(bytes) {
        if (::posix_memalign(reinterpret_cast<void**>(&p), align, bytes) != 0) {
            std::fprintf(stderr, "posix_memalign(%zu) failed\n", bytes);
            std::abort();
        }
    }
    ~AlignedBuf() { ::free(p); }
    uint8_t* data() { return p; }
    size_t   size() const { return n; }
    uint8_t& operator[](size_t i) { return p[i]; }
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constexpr uint32_t kNumShardsStd = 2;      // standard: K on dev0, V on dev1
static constexpr uint32_t kNumShardsMla = 1;      // MLA: single unified tensor
static constexpr uint32_t kDefaultBatch  = 500;    // BAR1-safe with 128KB tensors

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    uint64_t n_files     = 8000000;  // --n: 8M GpuFiles by default
    uint32_t batch_sz    = kDefaultBatch;
    uint64_t l1_mib      = 512;      // --l1-mib: L1 GPU budget
    uint64_t l2_mib      = 2048;     // --l2-mib: L2 host budget
    uint32_t n_layers    = 8;        // --layers: transformer layers
    uint32_t tensor_kb   = 128;      // --tensor-kb: per-tensor size
    uint64_t data_gb     = 0;        // --data-gb: total data transfer budget; 0 = unlimited (cover all files)
    uint32_t num_streams = 2;        // --streams: CUDA streams for mixed R/W
    uint32_t max_entries = 2048;     // --max-entries: io_engine batch depth (was 256; larger = fewer blocking syncs per layer)
    bool     use_mla     = false;    // --mla: unified single-tensor layout
    bool     shuffle_files = true;   // --no-shuffle to disable
    uint64_t shuffle_seed  = 42;     // --seed: RNG seed for access-order shuffle
    uint64_t chunk_size    = 500000; // --chunk-size: Phase1/4 batch-call chunk size (progress granularity)
    bool     skip_crash_consistency = false;  // --skip-crash-consistency
    bool     skip_delete            = false;  // --skip-delete
    bool     read_only             = false;  // --read-only: skip write, pure read (no verify)
    std::string log_path_override;            // --log <path>
    tutti_examples::RegistryCliOptions ropt;

    for (int i = 1; i < argc; ) {
        if (tutti_examples::parse_registry_cli_arg(ropt, argc, argv, i)) { ++i; continue; }
        const char* a = argv[i];
        if      (std::strcmp(a, "--n")        == 0 && i+1 < argc) { n_files     = std::strtoull(argv[++i], nullptr, 10); ++i; }
        else if (std::strcmp(a, "--batch")    == 0 && i+1 < argc) { batch_sz    = (uint32_t)std::strtoul(argv[++i], nullptr, 10); ++i; }
        else if (std::strcmp(a, "--l1-mib")   == 0 && i+1 < argc) { l1_mib      = std::strtoull(argv[++i], nullptr, 10); ++i; }
        else if (std::strcmp(a, "--l2-mib")   == 0 && i+1 < argc) { l2_mib      = std::strtoull(argv[++i], nullptr, 10); ++i; }
        else if (std::strcmp(a, "--layers")   == 0 && i+1 < argc) { n_layers    = (uint32_t)std::strtoul(argv[++i], nullptr, 10); ++i; }
        else if (std::strcmp(a, "--tensor-kb")== 0 && i+1 < argc) { tensor_kb   = (uint32_t)std::strtoul(argv[++i], nullptr, 10); ++i; }
        else if (std::strcmp(a, "--data-gb")  == 0 && i+1 < argc) { data_gb     = std::strtoull(argv[++i], nullptr, 10); ++i; }
        else if (std::strcmp(a, "--streams")  == 0 && i+1 < argc) { num_streams = (uint32_t)std::strtoul(argv[++i], nullptr, 10); ++i; }
        else if (std::strcmp(a, "--max-entries")== 0 && i+1 < argc) { max_entries = (uint32_t)std::strtoul(argv[++i], nullptr, 10); ++i; }
        else if (std::strcmp(a, "--mla")      == 0)                { use_mla     = true; ++i; }
        else if (std::strcmp(a, "--seed")     == 0 && i+1 < argc) { shuffle_seed = std::strtoull(argv[++i], nullptr, 10); ++i; }
        else if (std::strcmp(a, "--no-shuffle")== 0)               { shuffle_files = false; ++i; }
        else if (std::strcmp(a, "--chunk-size")== 0 && i+1 < argc) { chunk_size  = std::strtoull(argv[++i], nullptr, 10); ++i; }
        else if (std::strcmp(a, "--skip-crash-consistency") == 0)  { skip_crash_consistency = true; ++i; }
        else if (std::strcmp(a, "--skip-delete") == 0)             { skip_delete = true; ++i; }
        else if (std::strcmp(a, "--read-only") == 0)              { read_only = true; ++i; }
        else if (std::strcmp(a, "--log")      == 0 && i+1 < argc) { log_path_override = argv[++i]; ++i; }
        else { std::fprintf(stderr, "unknown arg: %s\n", a); return 1; }
    }
    if (!tutti_examples::validate_registry_cli(ropt, argv[0])) return 1;
    if (chunk_size == 0) chunk_size = 1;

    const uint64_t tensor_size     = (uint64_t)tensor_kb * 1024;
    const uint32_t num_shards      = use_mla ? kNumShardsMla : kNumShardsStd;
    const uint64_t gpu_file_total  = (uint64_t)num_shards * n_layers * tensor_size;
    if (num_streams == 0) num_streams = 1;

    // Open logfile at <repo_root>/kv_cache_e2e_stress.log.  This binary
    // lives at <repo_root>/build/bin/, so repo root is ../../ from cwd
    // when run from build/bin, but callers may run from anywhere --
    // resolve relative to this source file's known repo-root-relative
    // path is not possible at runtime, so we just use a fixed relative
    // path from the current working directory the smoke is invoked
    // from (documented in usage).  Fall back to /tmp on failure.
    {
        const char* log_path = log_path_override.empty()
            ? "./kv_cache_e2e_stress.log" : log_path_override.c_str();
        g_logfile = std::fopen(log_path, "w");
        if (g_logfile == nullptr) {
            g_logfile = std::fopen("/tmp/kv_cache_e2e_stress.log", "w");
        }
    }

    CUDA_OK(cudaFree(0));
    CUDA_OK(cudaSetDevice(ropt.cuda_dev));
    STEP_OK("cudaSetDevice(%d)", ropt.cuda_dev);

    // ---- Coordinator bootstrap ----
    tutti::CoordinatorConfig cfg;
    if (ropt.service_mode) {
        cfg.mode = tutti::CoordinatorMode::SERVICE_CLIENT;
        cfg.daemon_endpoint = ropt.service_endpoint;
        cfg.daemon_device_ids = ropt.dev_ids;
    } else {
        cfg.mode = tutti::CoordinatorMode::IN_PROCESS;
        cfg.pci_addrs = ropt.pci_addrs;
    }
    cfg.cuda_device = ropt.cuda_dev;
    cfg.num_user_queues_per_device = 4;
    cfg.max_entries_per_batch = max_entries;

    // L1/L2 budgets (configurable via --l1-mib/--l2-mib).  Default 128
    // MiB each.  Set both small (e.g. --l1-mib 4 --l2-mib 8) to force
    // heavy eviction; large (covering the whole working set) for the
    // no-eviction baseline.
    cfg.handle_l1_gpu_budget_bytes  = l1_mib * 1024 * 1024;
    cfg.handle_l2_host_budget_bytes = l2_mib * 1024 * 1024;

    tutti::Coordinator coord;
    if (!coord.bootstrap(cfg)) STEP_FAIL("Coordinator::bootstrap");
    STEP_OK("Coordinator bootstrap (L1=%lu MiB, L2=%lu MiB)",
            (unsigned long)l1_mib, (unsigned long)l2_mib);

    // Cache capacity in NvmeFile (shard) units.  sizeof(NvmeFileDeviceHandle)
    // = 192 B.  L1 (GPU-resident) is the binding tier for L1->L2 eviction;
    // L2 (CPU-pinned) is the binding tier for genuine L2 eviction (rebuild).
    const uint64_t cache_cap_files    = (l1_mib * 1024 * 1024) / 192;  // L1
    const uint64_t l2_cap_files       = (l2_mib * 1024 * 1024) / 192;  // L2

    auto coord_devs = coord.devices();
    if (coord_devs.size() < 2) STEP_FAIL("need >= 2 devices, got %zu", coord_devs.size());

    // Pre-cleanup: only remove stragglers that do NOT match our
    // canonical naming for the requested n_files -- if a prior run
    // already left exactly the file set we need, reuse it instead of
    // paying the create cost again.
    std::vector<tutti::GpuFileId> all_ids;
    bool need_create = true;
    {
        auto names = coord.block()->list_gpu_file_names();
        std::size_t matching = 0;
        for (const auto& nm : names)
            if (nm.rfind("kvs_", 0) == 0) ++matching;

        if (matching == n_files) {
            LOG_INFO("Pre-check: found existing %zu kvs_* GpuFiles matching --n; reusing (skip create)", matching);
            need_create = false;
        } else if (matching > 0) {
            LOG_INFO("Pre-cleanup: %zu existing kvs_* GpuFiles != requested %lu; removing",
                     matching, (unsigned long)n_files);
            std::vector<tutti::GpuFileSpec> specs;
            specs.reserve(matching);
            for (const auto& nm : names)
                if (nm.rfind("kvs_", 0) == 0) specs.push_back({nm});
            auto gfs = coord.open_gpu_files_batch(
                specs.data(), (uint32_t)specs.size(), tutti::GPU_FILE_OPEN_EXISTING);
            auto ok = std::make_unique<bool[]>(gfs.size());
            (void)coord.delete_gpu_files_batch(gfs.data(), (uint32_t)gfs.size(), ok.get());
            for (auto* d : coord_devs) coord.storage()->flush_metadata(d);
            coord.block()->flush_metadata();
        }
    }

    // ======================================================================
    // Phase 1: Batch create N GpuFiles (each 2 MB) via
    //          Coordinator::open_gpu_files_batch (multi-threaded internally).
    // ======================================================================
    if (need_create) {
        STEP_OK("Phase 1: batch create %lu GpuFiles (shape %ux%ux%luKB = %lu MB each), "
                "chunk-size=%lu — starting",
                (unsigned long)n_files, (unsigned)num_shards, (unsigned)n_layers,
                (unsigned long)tensor_kb, (unsigned long)(gpu_file_total / (1024*1024)),
                (unsigned long)chunk_size);

        all_ids.reserve(n_files);
        uint64_t n_failed = 0;
        auto t0 = std::chrono::steady_clock::now();
        auto t_last_log = t0;
        for (uint64_t off = 0; off < n_files; off += chunk_size) {
            uint64_t cnt = std::min(chunk_size, n_files - off);
            std::vector<std::string> names(cnt);
            std::vector<tutti::GpuFileSpec> specs(cnt);
            for (uint64_t j = 0; j < cnt; ++j) {
                uint64_t i = off + j;
                char nm[64];
                std::snprintf(nm, sizeof(nm), "kvs_%lu", (unsigned long)i);
                names[j] = nm;
                specs[j].name = names[j];
                specs[j].total_size = gpu_file_total;
                specs[j].tensor_shape[0] = num_shards;
                specs[j].tensor_shape[1] = n_layers;
                specs[j].tensor_shape[2] = (uint32_t)tensor_size;
                if (use_mla) {
                    specs[j].shard_placement = {coord_devs[0]};
                } else {
                    specs[j].shard_placement = {coord_devs[0], coord_devs[1]};
                }
            }
            auto results = coord.open_gpu_files_batch(
                specs.data(), (uint32_t)specs.size(),
                tutti::GPU_FILE_OPEN_CREATE | tutti::GPU_FILE_OPEN_NO_PERSIST);
            for (auto* gf : results) {
                if (gf == nullptr) { ++n_failed; continue; }
                all_ids.push_back(gf->id);
            }

            uint64_t done = off + cnt;
            if (seconds_since(t_last_log) >= kProgressHeartbeatSec || done >= n_files) {
                double elapsed = seconds_since(t0);
                double rate = elapsed > 0 ? (double)done / elapsed : 0.0;
                double eta = rate > 0 ? (double)(n_files - done) / rate : 0.0;
                LOG_INFO("Phase 1 progress: %lu/%lu created (%.1f%%), %.0f files/s, "
                         "elapsed=%s eta=%s",
                         (unsigned long)done, (unsigned long)n_files,
                         100.0 * (double)done / (double)n_files, rate,
                         format_duration(elapsed).c_str(), format_duration(eta).c_str());
                t_last_log = std::chrono::steady_clock::now();
            }
        }
        for (auto* d : coord_devs) coord.storage()->flush_metadata(d);
        coord.block()->flush_metadata();
        double sec = seconds_since(t0);

        if (n_failed != 0)
            STEP_FAIL("Phase 1: %lu/%lu GpuFile creates failed",
                      (unsigned long)n_failed, (unsigned long)n_files);

        STEP_OK("Phase 1: created %lu GpuFiles in %.3fs (%.0f files/s, %.1f MB/s)",
                (unsigned long)n_files, sec, (double)n_files / sec,
                (double)(n_files * gpu_file_total) / (1024.0*1024.0) / sec);
    } else {
        // Batch-open (multi-threaded, GPU_FILE_OPEN_EXISTING) instead
        // of a per-file open_gpu_file loop -- the serial loop was the
        // dominant cost at 4M-file scale (CUDA admit + FIEMAP-miss
        // overhead per call, single-threaded).  Chunked the same way
        // as the create path so a reuse of an 8M-file set still logs
        // progress instead of blocking silently on one giant call.
        auto names = coord.block()->list_gpu_file_names();
        std::vector<tutti::GpuFileSpec> specs;
        specs.reserve(names.size());
        for (const auto& nm : names) {
            if (nm.rfind("kvs_", 0) != 0) continue;
            specs.push_back({nm});
        }
        const uint64_t total = specs.size();
        all_ids.reserve(total);
        uint64_t n_failed = 0;
        auto t0 = std::chrono::steady_clock::now();
        auto t_last_log = t0;
        for (uint64_t off = 0; off < total; off += chunk_size) {
            uint64_t cnt = std::min(chunk_size, total - off);
            auto results = coord.open_gpu_files_batch(
                specs.data() + off, (uint32_t)cnt, tutti::GPU_FILE_OPEN_EXISTING);
            for (auto* gf : results) {
                if (gf == nullptr) { ++n_failed; continue; }
                all_ids.push_back(gf->id);
            }
            uint64_t done = off + cnt;
            if (seconds_since(t_last_log) >= kProgressHeartbeatSec || done >= total) {
                double elapsed = seconds_since(t0);
                double rate = elapsed > 0 ? (double)done / elapsed : 0.0;
                double eta = rate > 0 ? (double)(total - done) / rate : 0.0;
                LOG_INFO("Phase 1 progress: %lu/%lu reused (%.1f%%), %.0f files/s, "
                         "elapsed=%s eta=%s",
                         (unsigned long)done, (unsigned long)total,
                         100.0 * (double)done / (double)total, rate,
                         format_duration(elapsed).c_str(), format_duration(eta).c_str());
                t_last_log = std::chrono::steady_clock::now();
            }
        }
        double sec = seconds_since(t0);

        if (n_failed != 0)
            STEP_FAIL("Phase 1: %lu/%lu GpuFile re-opens failed",
                      (unsigned long)n_failed, (unsigned long)total);
        STEP_OK("Phase 1: reused %zu existing GpuFiles in %.3fs (%.0f files/s, skip create)",
                all_ids.size(), sec, (double)all_ids.size() / sec);
    }
    // Verify count.
    {
        auto names = coord.block()->list_gpu_file_names();
        std::size_t matching = 0;
        for (const auto& nm : names) if (nm.rfind("kvs_", 0) == 0) ++matching;
        if (matching != n_files)
            STEP_FAIL("Phase 1: count mismatch: expected %lu, got %zu",
                      (unsigned long)n_files, matching);
        STEP_OK("Phase 1: verified %zu GpuFiles in directory", matching);
    }

    // Randomize the GpuFileId access order used by Phase 2 (default
    // on; --no-shuffle to disable) -- exercises a non-sequential SSD
    // access pattern instead of always reading kvs_0..kvs_N in
    // creation order.  Safe to shuffle: each batch's write is
    // immediately followed by that SAME batch's read (same fids, same
    // stream), so per-position seed verification is unaffected by
    // which physical file sits at a given position.
    std::vector<tutti::GpuFileId> rw_order = all_ids;
    if (shuffle_files) {
        std::mt19937_64 rng(shuffle_seed);
        std::shuffle(rw_order.begin(), rw_order.end(), rng);
        STEP_OK("Phase 2: shuffled %zu GpuFileIds for randomized access order (seed=%lu)",
                rw_order.size(), (unsigned long)shuffle_seed);
    }

    // ======================================================================
    // Phase 2: Multi-stream, multi-layer GPU R/W via KvCacheIoAdapter.
    //
    // D-mla + D-stream + D-scale combined:
    //   - n_layers transformer layers (default 8), each with K+V tensors
    //   - num_streams CUDA streams alternating across batches (tests
    //     cross-stream handle-cache correctness via GpuSlotPool::wait_ready)
    //   - data_gb budget caps total transfer (we create n_files for
    //     metadata scale but only R/W a data-sized subset)
    //   - Per-layer, per-K/V verification: seed encodes (file, layer, K/V)
    //     so cross-layer / cross-K-V contamination is caught.
    // ======================================================================
    std::vector<cudaStream_t> streams(num_streams);
    for (auto& s : streams) CUDA_OK(cudaStreamCreate(&s));
    STEP_OK("Phase 2: created %u CUDA streams", (unsigned)num_streams);

    tutti::KvCacheIoAdapter kv(coord, tensor_size, use_mla);

    // Allocate + register device tensors ONCE (reused across batches
    // and layers).  Standard: K+V per slot; MLA: single tensor per slot.
    const uint32_t tensors_per_block = use_mla ? 1u : 2u;
    const uint32_t n_blocks = (uint32_t)std::min((uint64_t)batch_sz, n_files);
    std::vector<tutti::MemoryRegion*> k_regs(n_blocks, nullptr);  // K (or unified)
    std::vector<tutti::MemoryRegion*> v_regs(n_blocks, nullptr);  // V (unused for MLA)
    for (uint32_t b = 0; b < n_blocks; ++b) {
        k_regs[b] = coord.allocate_device(tensor_size, tutti::MemoryKind::DEVICE, ropt.cuda_dev);
        if (!k_regs[b]) STEP_FAIL("Phase 2: allocate_device K #%u", b);
        tutti::TensorRegistrationSpec ds{};
        ds.ptr = k_regs[b]->device_ptr; ds.size = tensor_size; ds.granularity = tensor_size;
        if (coord.register_tensor(ds) == nullptr)
            STEP_FAIL("Phase 2: register_tensor(K #%u) failed after ~%.0f MiB "
                      "BAR1 pinned -- GPU BAR1 P2P window exhausted; lower "
                      "--batch (currently %u) or --tensor-kb (T1)",
                      b, (double)(b * tensors_per_block * (tensor_size + 131072)) / (1024.0*1024.0),
                      batch_sz);
        if (!use_mla) {
            v_regs[b] = coord.allocate_device(tensor_size, tutti::MemoryKind::DEVICE, ropt.cuda_dev);
            if (!v_regs[b]) STEP_FAIL("Phase 2: allocate_device V #%u", b);
            ds.ptr = v_regs[b]->device_ptr;
            if (coord.register_tensor(ds) == nullptr)
                STEP_FAIL("Phase 2: register_tensor(V #%u) failed -- BAR1 window "
                          "exhausted; lower --batch (currently %u) (T1)", b, batch_sz);
        }
    }
    STEP_OK("Phase 2: allocated + registered %u %s tensors (%.2f GiB GPU)",
            n_blocks * tensors_per_block, use_mla ? "unified" : "K+V",
            (double)(n_blocks * tensors_per_block * tensor_size) / (1024.0*1024.0*1024.0));

    // How many files get actual data R/W.  --data-gb 0 (default) means
    // unlimited -- cover every created file (metadata scale == data
    // scale); a nonzero --data-gb caps the covered subset instead.
    const uint64_t bytes_per_file = gpu_file_total;  // all layers, all shards
    const uint64_t n_data_files = (data_gb == 0)
        ? n_files
        : std::min(n_files, (data_gb * (uint64_t)(1ull << 30)) / bytes_per_file);
    const uint64_t n_batches = (n_data_files + n_blocks - 1) / n_blocks;
    const uint64_t total_rw_bytes = n_data_files * bytes_per_file * 2;  // write + read

    if (data_gb == 0) {
        LOG_INFO("Phase 2: %lu files created, ALL get data R/W (unlimited --data-gb), "
                 "%lu batches of %u, ~%.2f TB total R/W", (unsigned long)n_files,
                 (unsigned long)n_batches, (unsigned)n_blocks,
                 (double)total_rw_bytes / (1024.0*1024.0*1024.0*1024.0));
    } else {
        LOG_INFO("Phase 2: %lu files created, %lu files get data R/W (%lu GB budget, "
                 "%lu batches of %u, ~%.2f TB total R/W)", (unsigned long)n_files,
                 (unsigned long)n_data_files, (unsigned long)data_gb,
                 (unsigned long)n_batches, (unsigned)n_blocks,
                 (double)total_rw_bytes / (1024.0*1024.0*1024.0*1024.0));
    }

    uint64_t total_verified = 0;
    uint64_t total_mismatch = 0;
    uint64_t total_bytes_written = 0;
    uint64_t total_bytes_read = 0;

    void* h_verify = nullptr;
    CUDA_OK(cudaMallocHost(&h_verify, tensor_size));

    tutti::INvmeStorage::CacheStats cs_before = coord.cache_stats();

    double sec_write = 0.0, sec_read = 0.0;
    auto t0 = std::chrono::steady_clock::now();
    auto t_last_log = t0;

    for (uint64_t batch = 0; batch < n_batches; ++batch) {
        uint64_t bstart = batch * n_blocks;
        uint64_t bend   = std::min(bstart + n_blocks, n_data_files);
        uint32_t cnt    = (uint32_t)(bend - bstart);
        cudaStream_t s  = streams[batch % num_streams];  // alternate streams

        std::vector<tutti::GpuFileId> fids(rw_order.begin() + bstart,
                                            rw_order.begin() + bend);
        std::vector<tutti::MemoryRegion*> k_batch(k_regs.begin(), k_regs.begin() + cnt);
        std::vector<tutti::MemoryRegion*> v_batch(v_regs.begin(), v_regs.begin() + cnt);

        // --- Write all layers (skipped in --read-only mode) ---
        if (!read_only) {
        // The GPU-side fill (cudaMemsetAsync) is buffer prep, NOT part of
        // the NVMe write path, so it is done+synced BEFORE starting the
        // write timer -- counting it would understate write-interface
        // bandwidth.  sec_write therefore measures only batched_write
        // (the GPU->NVMe path) + the sync that completes it.
        for (uint32_t L = 0; L < n_layers; ++L) {
            // Fill with layer-specific seed: K and V get different bytes.
            for (uint32_t b = 0; b < cnt; ++b) {
                uint8_t seed_k = (uint8_t)(((bstart + b) * n_layers + L) & 0xFF);
                CUDA_OK(cudaMemsetAsync(k_batch[b]->device_ptr, seed_k, tensor_size, s));
                if (!use_mla) {
                    uint8_t seed_v = (uint8_t)(seed_k + 0x80);
                    CUDA_OK(cudaMemsetAsync(v_batch[b]->device_ptr, seed_v, tensor_size, s));
                }
            }
            CUDA_OK(cudaStreamSynchronize(s));

            auto tw0 = std::chrono::steady_clock::now();
            if (use_mla) {
                if (!kv.batched_write_unified((int)L, k_batch, fids, s))
                    STEP_FAIL("Phase 2: batched_write_unified L=%u batch=%lu",
                              L, (unsigned long)batch);
            } else {
                if (!kv.batched_write((int)L, k_batch, v_batch, fids, s))
                    STEP_FAIL("Phase 2: batched_write L=%u batch=%lu",
                              L, (unsigned long)batch);
            }
            CUDA_OK(cudaStreamSynchronize(s));
            sec_write += seconds_since(tw0);
        }
        total_bytes_written += (uint64_t)cnt * tensors_per_block * n_layers * tensor_size;

        // Durability point for this batch.
        for (auto fid : fids) coord.sync_file(fid, s);
        } // if (!read_only)

        // --- Read + verify all layers ---
        // Both the pre-clear (cudaMemsetAsync, to prove the read actually
        // refills the buffers) AND the verification are OUTSIDE the read
        // timer.  The verification does one D2H cudaMemcpyAsync + a full
        // cudaStreamSynchronize PER block PER K/V PER layer -- thousands
        // of serial GPU round-trips per batch that dwarf the NVMe read
        // itself; timing them together previously understated read
        // bandwidth by ~an order of magnitude.  sec_read now measures
        // only batched_read (the NVMe->GPU path) + its completing sync.
        for (uint32_t L = 0; L < n_layers; ++L) {
            // Clear tensors (skip in --read-only: no verification means
            // the pre-clear is unnecessary overhead).
            if (!read_only) {
            for (uint32_t b = 0; b < cnt; ++b) {
                CUDA_OK(cudaMemsetAsync(k_batch[b]->device_ptr, 0, tensor_size, s));
                if (!use_mla)
                    CUDA_OK(cudaMemsetAsync(v_batch[b]->device_ptr, 0, tensor_size, s));
            }
            CUDA_OK(cudaStreamSynchronize(s));
            }

            auto tr0 = std::chrono::steady_clock::now();
            if (use_mla) {
                if (!kv.batched_read_unified((int)L, k_batch, fids, s))
                    STEP_FAIL("Phase 2: batched_read_unified L=%u batch=%lu",
                              L, (unsigned long)batch);
            } else {
                if (!kv.batched_read((int)L, k_batch, v_batch, fids, s))
                    STEP_FAIL("Phase 2: batched_read L=%u batch=%lu",
                              L, (unsigned long)batch);
            }
            CUDA_OK(cudaStreamSynchronize(s));
            sec_read += seconds_since(tr0);

            // Verify K (and V) for this layer (untimed -- see note above).
            // Skipped in --read-only: files may not have been written by
            // this process, so seed-based verification is meaningless.
            if (!read_only) {
            for (uint32_t b = 0; b < cnt; ++b) {
                uint8_t expect_k = (uint8_t)(((bstart + b) * n_layers + L) & 0xFF);
                CUDA_OK(cudaMemcpyAsync(h_verify, k_batch[b]->device_ptr,
                                        tensor_size, cudaMemcpyDeviceToHost, s));
                CUDA_OK(cudaStreamSynchronize(s));
                if (((uint8_t*)h_verify)[0] != expect_k) {
                    ++total_mismatch;
                    if (total_mismatch == 1)
                        LOG_INFO("K mismatch batch=%lu b=%u L=%u expected=0x%02x got=0x%02x "
                                 "stream=%u",
                                 (unsigned long)batch, b, L, (unsigned)expect_k,
                                 (unsigned)((uint8_t*)h_verify)[0],
                                 (unsigned)(batch % num_streams));
                }
                if (!use_mla) {
                    uint8_t expect_v = (uint8_t)(expect_k + 0x80);
                    CUDA_OK(cudaMemcpyAsync(h_verify, v_batch[b]->device_ptr,
                                            tensor_size, cudaMemcpyDeviceToHost, s));
                    CUDA_OK(cudaStreamSynchronize(s));
                    if (((uint8_t*)h_verify)[0] != expect_v) {
                        ++total_mismatch;
                        if (total_mismatch == 1)
                            LOG_INFO("V mismatch batch=%lu b=%u L=%u expected=0x%02x got=0x%02x "
                                     "stream=%u",
                                     (unsigned long)batch, b, L, (unsigned)expect_v,
                                     (unsigned)((uint8_t*)h_verify)[0],
                                     (unsigned)(batch % num_streams));
                    }
                }
                ++total_verified;
            }
            } // if (!read_only)  [verify]
        }
        total_bytes_read += (uint64_t)cnt * tensors_per_block * n_layers * tensor_size;

        bool time_trigger = seconds_since(t_last_log) >= kProgressHeartbeatSec;
        if (batch % 20 == 0 || batch == n_batches - 1 || time_trigger) {
            double elapsed = seconds_since(t0);
            double rate_batches = elapsed > 0 ? (double)(batch + 1) / elapsed : 0.0;
            double eta = rate_batches > 0 ? (double)(n_batches - (batch + 1)) / rate_batches : 0.0;
            double gb_done = (double)(total_bytes_written + total_bytes_read)
                              / (1024.0*1024.0*1024.0);
            LOG_INFO("Phase 2 progress: [batch %lu/%lu, %.1f%%] verified=%lu mismatches=%lu "
                     "stream=%u elapsed=%s eta=%s rw=%.1fGB",
                     (unsigned long)(batch+1), (unsigned long)n_batches,
                     100.0 * (double)(batch+1) / (double)n_batches,
                     (unsigned long)total_verified, (unsigned long)total_mismatch,
                     (unsigned)(batch % num_streams), format_duration(elapsed).c_str(),
                     format_duration(eta).c_str(), gb_done);
            t_last_log = std::chrono::steady_clock::now();
        }
    }

    double sec_total = seconds_since(t0);
    if (total_mismatch != 0)
        STEP_FAIL("Phase 2: %lu data mismatches in %lu tensor-reads",
                  (unsigned long)total_mismatch, (unsigned long)total_verified);

    // Cache-activity delta for THIS phase.
    tutti::INvmeStorage::CacheStats cs_after = coord.cache_stats();
    const uint64_t d_cold = cs_after.cold_builds   - cs_before.cold_builds;
    const uint64_t d_l1h  = cs_after.l1_hits       - cs_before.l1_hits;
    const uint64_t d_l2h  = cs_after.l2_hits       - cs_before.l2_hits;
    const uint64_t d_prom = cs_after.l1_promotions - cs_before.l1_promotions;
    const uint64_t d_l1e  = cs_after.l1_evictions  - cs_before.l1_evictions;
    const uint64_t d_l2e  = cs_after.l2_evictions  - cs_before.l2_evictions;
    // working set = shards touched (num_shards per GpuFile); cache cap is
    // in NvmeFile (shard) units.
    const uint64_t working_set_shards = (uint64_t)n_data_files * num_shards;
    const bool eviction_expected = working_set_shards > (uint64_t)cache_cap_files;

    // Per-interface throughput.  sec_write / sec_read now each isolate
    // ONLY the batched_write / batched_read NVMe path (fill + D2H verify
    // excluded, see the timed regions above), so these numbers reflect
    // the write and read interfaces separately.  K and V shards are
    // placed on dev0 / dev1 respectively (non-MLA), so a single logical
    // R/W drives both NVMe devices in parallel -- expect aggregate
    // (2-drive) bandwidth, not single-drive.
    const double gib = 1024.0 * 1024.0 * 1024.0;
    const double write_gbs = (double)total_bytes_written / gib / sec_write;
    const double read_gbs  = (double)total_bytes_read    / gib / sec_read;
    STEP_OK("Phase 2 THROUGHPUT: %lu files x %u layers, %u streams, batch=%u "
            "(concurrency) — total wall=%.3fs%s",
            (unsigned long)n_data_files, (unsigned)n_layers, (unsigned)num_streams,
            (unsigned)n_blocks, sec_total, read_only ? " [READ-ONLY]" : "");
    if (!read_only)
    STEP_OK("Phase 2   WRITE interface: %.3fs, %.2f GiB written, %.3f GiB/s "
            "(%.1f MB/s)  [GPU->NVMe, batched_write only]",
            sec_write, (double)total_bytes_written / gib, write_gbs, write_gbs * 1024.0);
    STEP_OK("Phase 2   READ  interface: %.3fs, %.2f GiB read, %.3f GiB/s "
            "(%.1f MB/s)  [NVMe->GPU, batched_read only]",
            sec_read, (double)total_bytes_read / gib, read_gbs, read_gbs * 1024.0);

    // Metadata (handle) cache state + activity.  cap is in NvmeFile
    // (shard) units; working_set = shards actually touched this phase.
    const uint64_t d_lookups = d_cold + d_l1h + d_l2h;
    const double   l1_hit_pct = d_lookups ? 100.0 * (double)d_l1h  / (double)d_lookups : 0.0;
    const double   l2_hit_pct = d_lookups ? 100.0 * (double)d_l2h  / (double)d_lookups : 0.0;
    const double   cold_pct   = d_lookups ? 100.0 * (double)d_cold / (double)d_lookups : 0.0;
    STEP_OK("Phase 2 CACHE STATE [%s]: L1 cap=%lu shards (%lu MiB), "
            "L2 cap=%lu shards (%lu MiB), working_set=%lu shards",
            eviction_expected ? "L1-EVICTION regime" : "NO-EVICTION regime",
            (unsigned long)cache_cap_files, (unsigned long)l1_mib,
            (unsigned long)l2_cap_files,   (unsigned long)l2_mib,
            (unsigned long)working_set_shards);
    STEP_OK("Phase 2 CACHE ACTIVITY: lookups=%lu -> L1_hits=%lu (%.1f%%) "
            "L2_hits=%lu (%.1f%%) cold_builds=%lu (%.1f%%) | "
            "L1_promotions(L2->L1)=%lu L1_evictions(L1->L2)=%lu "
            "L2_evictions(delete)=%lu",
            (unsigned long)d_lookups,
            (unsigned long)d_l1h, l1_hit_pct,
            (unsigned long)d_l2h, l2_hit_pct,
            (unsigned long)d_cold, cold_pct,
            (unsigned long)d_prom, (unsigned long)d_l1e, (unsigned long)d_l2e);
    if (eviction_expected && d_l1e == 0)
        STEP_FAIL("Phase 2: working set (%lu) > cache cap (%lu) but 0 L1 "
                  "evictions -- eviction path not exercised",
                  (unsigned long)working_set_shards, (unsigned long)cache_cap_files);

    for (uint32_t b = 0; b < n_blocks; ++b) {
        if (k_regs[b]) coord.free(k_regs[b]);
        if (v_regs[b]) coord.free(v_regs[b]);
    }
    CUDA_OK(cudaFreeHost(h_verify));
    for (auto& s : streams) CUDA_OK(cudaStreamDestroy(s));

    // ======================================================================
    // Phase 3: Crash consistency (mirrors nvme_storage_e2e_stress.cu).
    // Skippable with --skip-crash-consistency (e.g. sweep runs that
    // reuse the file set across --batch/cache-size variants and only
    // want the LAST run in the sweep to pay this cost).
    // ======================================================================
    if (skip_crash_consistency) {
    STEP_OK("Phase 3: SKIPPED (--skip-crash-consistency)");
    } else {

    // ---- 3a: Ghost sweep ----
    // Stray shard .bin files created directly on the host filesystem
    // (bypassing the storage API -> no log entry).  Re-bootstrap ->
    // nvme_storage's reconcile pass must unlink them.  mount_path() is
    // a documented Coordinator/INvmeStorage accessor (not "internal"),
    // used here because there is no adapter-level equivalent for
    // fault injection.
    STEP_OK("Phase 3a: ghost sweep — starting");
    {
        const uint32_t n_ghosts = 1000;
        std::string mp = coord.storage()->mount_path(coord_devs[0]);
        std::string tutti_dir = mp + "/.tutti";
        for (uint32_t i = 0; i < n_ghosts; ++i) {
            char nm[96];
            std::snprintf(nm, sizeof(nm), "%s/ghost_%u.bin", tutti_dir.c_str(), i);
            int fd = ::open(nm, O_CREAT | O_RDWR | O_CLOEXEC, 0600);
            if (fd >= 0) { ::fallocate(fd, 0, 0, 4096); ::close(fd); }
        }
        std::size_t before = coord.block()->list_gpu_file_names().size();

        coord.shutdown();
        if (!coord.bootstrap(cfg)) STEP_FAIL("Phase 3a: re-bootstrap");

        std::size_t after = coord.block()->list_gpu_file_names().size();
        if (after != before)
            STEP_FAIL("Phase 3a: ghost sweep: before=%zu after=%zu", before, after);
        STEP_OK("Phase 3a: %u ghosts cleaned by reconcile (count stable at %zu)", n_ghosts, after);
    }

    // ---- 3b: External rm ----
    // `rm` a shard's original .bin path directly (simulating another
    // process misusing the filesystem).  The .refs/ hardlink keeps the
    // inode alive; log-side, the shard becomes a tombstone that
    // reconcile drops on the next bootstrap.
    STEP_OK("Phase 3b: external rm — starting");
    {
        auto names = coord.block()->list_gpu_file_names();
        const uint32_t n_rm = std::max(1u, std::min((uint32_t)100,
            (uint32_t)(names.size() / 10)));  // at most 10%, leave rest for 3c
        if (n_rm == 0) STEP_FAIL("Phase 3b: no files");

        for (uint32_t i = 0; i < n_rm; ++i) {
            auto* gf = coord.open_gpu_file({names[i]});
            if (!gf) continue;
            for (auto* shard : gf->shards) ::unlink(shard->host_path.c_str());
        }
        std::size_t before = coord.block()->list_gpu_file_names().size();

        coord.shutdown();
        if (!coord.bootstrap(cfg)) STEP_FAIL("Phase 3b: re-bootstrap");

        std::size_t after = coord.block()->list_gpu_file_names().size();
        STEP_OK("Phase 3b: external rm — before=%zu after=%zu (reconcile cleaned tombstones)",
                before, after);
    }

    // ---- 3c: Data durability bounce ----
    STEP_OK("Phase 3c: data durability bounce — starting");
    {
        auto names = coord.block()->list_gpu_file_names();
        const uint32_t n_bounce = std::min((uint32_t)100, (uint32_t)names.size());
        if (n_bounce == 0) STEP_FAIL("Phase 3c: no files");

        AlignedBuf buf(tensor_size);
        for (uint32_t i = 0; i < n_bounce; ++i) {
            auto* gf = coord.open_gpu_file({names[i]});
            if (!gf) STEP_FAIL("Phase 3c: open(%s)", names[i].c_str());
            uint8_t seed = (uint8_t)(0xD0 + i);
            std::memset(buf.data(), seed, buf.size());
            for (auto* shard : gf->shards) {
                coord.storage()->write_blocking(shard, 0, buf.data(), tensor_size);
                coord.storage()->sync(shard);
            }
        }

        coord.shutdown();
        if (!coord.bootstrap(cfg)) STEP_FAIL("Phase 3c: re-bootstrap");

        uint32_t mismatch = 0;
        for (uint32_t i = 0; i < n_bounce; ++i) {
            auto* gf = coord.open_gpu_file({names[i]});
            if (!gf) STEP_FAIL("Phase 3c: reopen(%s)", names[i].c_str());
            uint8_t seed = (uint8_t)(0xD0 + i);
            for (auto* shard : gf->shards) {
                std::memset(buf.data(), 0, buf.size());
                coord.storage()->read_blocking(shard, 0, buf.data(), tensor_size);
                if (buf[0] != seed) { ++mismatch; break; }
            }
        }
        if (mismatch != 0)
            STEP_FAIL("Phase 3c: %u/%u files mismatch after bounce", mismatch, n_bounce);
        STEP_OK("Phase 3c: %u files verified data-durable across bounce", n_bounce);
    }
    } // if (!skip_crash_consistency)

    // ======================================================================
    // Phase 4: Bulk delete (batch open + batch delete, both threaded
    // in nvme_storage -- replaces the old serial open+delete loop
    // that was the dominant cost at multi-million-file scale).
    // Chunked into --chunk-size groups for progress visibility;
    // skippable with --skip-delete to leave the file set in place for
    // the next sweep invocation (--batch/--l1-mib/--l2-mib variant).
    // ======================================================================
    if (skip_delete) {
        STEP_OK("Phase 4: SKIPPED (--skip-delete) — %zu GpuFiles left in place for reuse",
                coord.block()->list_gpu_file_names().size());
    } else {
        STEP_OK("Phase 4: bulk delete — starting");
        auto names = coord.block()->list_gpu_file_names();
        const std::size_t total = names.size();
        std::size_t n_failed = 0;
        auto td0 = std::chrono::steady_clock::now();
        auto t_last_log = td0;

        for (std::size_t off = 0; off < total; off += chunk_size) {
            std::size_t cnt = std::min((std::size_t)chunk_size, total - off);
            std::vector<tutti::GpuFileSpec> del_specs;
            del_specs.reserve(cnt);
            for (std::size_t k = 0; k < cnt; ++k) del_specs.push_back({names[off + k]});
            auto gfs = coord.open_gpu_files_batch(
                del_specs.data(), (uint32_t)del_specs.size(), tutti::GPU_FILE_OPEN_EXISTING);

            auto del_ok = std::make_unique<bool[]>(gfs.size());
            (void)coord.delete_gpu_files_batch(gfs.data(), (uint32_t)gfs.size(), del_ok.get());
            for (std::size_t k = 0; k < gfs.size(); ++k) if (!del_ok[k]) ++n_failed;

            std::size_t done = off + cnt;
            if (seconds_since(t_last_log) >= kProgressHeartbeatSec || done >= total) {
                double elapsed = seconds_since(td0);
                double rate = elapsed > 0 ? (double)done / elapsed : 0.0;
                double eta = rate > 0 ? (double)(total - done) / rate : 0.0;
                LOG_INFO("Phase 4 progress: %zu/%zu deleted (%.1f%%), %.0f files/s, "
                         "elapsed=%s eta=%s", done, total, 100.0 * (double)done / (double)total,
                         rate, format_duration(elapsed).c_str(), format_duration(eta).c_str());
                t_last_log = std::chrono::steady_clock::now();
            }
        }
        for (auto* d : coord_devs) coord.storage()->flush_metadata(d);
        coord.block()->flush_metadata();
        double sec = seconds_since(td0);
        if (n_failed != 0)
            STEP_FAIL("Phase 4: %zu/%zu GpuFile deletes failed", n_failed, total);
        std::size_t after = coord.block()->list_gpu_file_names().size();
        if (after != 0) STEP_FAIL("Phase 4: %zu files remain", after);
        STEP_OK("Phase 4: deleted %zu GpuFiles in %.3fs (%.0f files/s)",
                total, sec, (double)total / sec);
    }

    coord.shutdown();
    if (skip_crash_consistency || skip_delete) {
        char summary[256];
        std::snprintf(summary, sizeof(summary),
                       "\n=== kv_cache_e2e_stress: RW-ONLY PASSED (skipped: %s%s%s) ===",
                       skip_crash_consistency ? "crash-consistency" : "",
                       (skip_crash_consistency && skip_delete) ? "+" : "",
                       skip_delete ? "delete" : "");
        log_line("", summary);
    } else {
        log_line("", "\n=== kv_cache_e2e_stress: ALL PHASES PASSED ===");
    }
    if (g_logfile) std::fclose(g_logfile);
    return 0;
}
