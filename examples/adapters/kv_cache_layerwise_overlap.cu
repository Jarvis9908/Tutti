/**
 * kv_cache_layerwise_overlap.cu -- layerwise KV-cache R/W with
 * 3-stream compute/read/write overlap, simulating an HY3-shaped
 * 128K-context request at a 90% prefix-cache hit rate.
 *
 * Model
 * -----
 *   - One request = --ctx-tokens (default 131072 = 128K) split into
 *     --chunk-tokens (default 256) chunks -> 512 KV chunks, ONE GpuFile
 *     per chunk ("kvlw_<i>"), laid out 2 shards (K,V) x --layers
 *     (default 80) x --tensor-kb (default 128).  K/V shard pairs are
 *     placed round-robin across ALL attached devices ({dev0,dev1},
 *     {dev2,dev3}, ...), so every batch drives every NVMe at once.
 *   - --hit-pct (default 90) of the chunks are HIT: their KV is
 *     pre-written to SSD during setup, then READ back layer-by-layer
 *     in the simulated serving pass.  The remaining ~10% are MISS:
 *     "recomputed" per layer (a real SGEMM kernel on the compute
 *     stream) and WRITTEN back layer-by-layer.
 *   - Three CUDA streams: s_read (prefetch layer L+1), s_compute
 *     (layer L math), s_write (store layer L-1).  Steady-state
 *     pipeline per layer:  read(L+1) || compute(L) || write(L-1).
 *     Cross-stream deps go through cudaEvents; the load-bearing one
 *     is write(L) waiting on compute(L) (store only what was
 *     produced).  compute(L) additionally waits on read(L) to model
 *     "hit data arrives before the layer runs".
 *
 * Why host-serial submits still overlap on GPU
 * --------------------------------------------
 *   KvCacheIoAdapter::batched_* BLOCKS the host until its stream
 *   drains (submit_chunked -> Coordinator::submit_batch ->
 *   cudaStreamSynchronize), and the io_engine owns ONE shared
 *   d_scratch_ entry buffer that must never be raced.  So all IO
 *   submits are issued from this single host thread; the blocking
 *   semantics double as the scratch-safety mechanism.  The compute
 *   stream, however, is fed with plain async kernel launches that
 *   return immediately -- so while the host sits inside one
 *   batched_read/batched_write, the compute kernel for the current
 *   layer runs concurrently on the GPU.  Overlap comes from the
 *   stream/event graph, not from concurrent submits.  read-vs-write
 *   IO concurrency is deliberately NOT attempted (shared scratch).
 *
 * Balanced compute vs data movement
 * ---------------------------------
 *   --compute-us 0 (default) auto-calibrates the per-layer compute
 *   time to the measured per-layer IO time (one layer's read+write),
 *   so the pipeline is bubble-free; set --compute-us explicitly to
 *   model a faster/slower inference engine.
 *
 * Usage (SERVICE_CLIENT; daemon already running per sys_config.yaml):
 *   sudo ./bin/kv_cache_layerwise_overlap --cuda 0 \
 *       --service 127.0.0.1:50051 \
 *       --dev-id 0 --dev-id 1 --dev-id 2 --dev-id 3 --queues 32 \
 *       [--layers 80] [--ctx-tokens 131072] [--chunk-tokens 256] \
 *       [--hit-pct 90] [--tensor-kb 512] [--requests 2] \
 *       [--compute-us 0] [--gemm-n 1024] [--compute-sms 64] \
 *       [--l1-mib 512] [--l2-mib 2048] [--max-entries 8192] \
 *       [--skip-create] [--recreate] [--skip-delete] [--verify]
 *
 * nsys (records ONLY the simulated pass, thanks to cudaProfilerStart):
 *   sudo nsys profile -c cudaProfilerApi -t cuda,nvtx,osrt \
 *       --gpu-metrics-device=0 --cuda-memory-usage=true \
 *       -o tutti_layerwise --force-overwrite=true \
 *       ./bin/kv_cache_layerwise_overlap --cuda 0 \
 *       --service 127.0.0.1:50051 \
 *       --dev-id 0 --dev-id 1 --dev-id 2 --dev-id 3 --queues 32 \
 *       --layers 80 --requests 2
 *
 *   What to look at in the timeline:
 *     - CUDA kernels row: nvme_batch_xfer_kernel (IO, on the read /
 *       write streams) overlapping sgemm_layer_kernel (compute
 *       stream); NVTX ranges mark per-layer read/compute/write spans.
 *     - --gpu-metrics-device=0: PCIe RX/TX throughput (NVMe P2P DMA
 *       crosses the GPU's PCIe link).
 *     - osrt row: CUDA API calls (cudaMemcpyAsync / cudaLaunchKernel /
 *       cudaStreamSynchronize / cudaStreamWaitEvent) on the CPU side.
 */

#include "../../coordinator/include/coordinator.h"
#include "../../coordinator/include/coordinator_config.h"
#include "../../coordinator/include/device.h"
#include "../../block_storage/include/block_storage.h"
#include "../../block_storage/include/host_fs_backed_block_storage.h"
#include "../../nvme_storage/include/host_fs_backed_nvme_storage.h"
#include "../../memory/include/memory_region.h"
#include "kv_cache_io_adapter.h"

#include "../common/registry_cli.h"

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------

#define STEP_OK(...) do { \
    char _buf[1024]; std::snprintf(_buf, sizeof(_buf), __VA_ARGS__); \
    std::fprintf(stderr, "[ OK ] %s\n", _buf); \
} while (0)

#define STEP_FAIL(...) do { \
    char _buf[1024]; std::snprintf(_buf, sizeof(_buf), __VA_ARGS__); \
    std::fprintf(stderr, "[FAIL] %s\n", _buf); \
    std::_Exit(2); \
} while (0)

#define LOG_INFO(...) do { \
    char _buf[1024]; std::snprintf(_buf, sizeof(_buf), __VA_ARGS__); \
    std::fprintf(stderr, "[INFO] %s\n", _buf); \
} while (0)

#define CUDA_OK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) STEP_FAIL("CUDA: %s (%s)", #call, cudaGetErrorString(_e)); \
} while (0)

static double seconds_since(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// Stand-in for one transformer layer's math: a REAL SGEMM kernel
// (C = A x B, n x n, fp32), iterated `iters` times inside one launch so
// each layer shows up as a single dense kernel on the compute stream.
// Unlike a busy-wait, this produces genuine FLOPs and HBM traffic, so
// nsys GPU metrics show realistic SM / memory-bandwidth activity.
//
// GRID-STRIDE with a deliberately SMALL grid (--compute-sms blocks):
// a full-matrix grid (e.g. 16384 blocks for n=2048) saturates every SM
// on the GPU until the kernel retires -- the nvme_batch_xfer_kernel on
// the IO streams can then only be scheduled into the gaps BETWEEN
// compute kernels, which is exactly the "no overlap, PCIe idles at a
// few GB/s" failure mode.  Capping the grid leaves free SMs (and free
// block slots on the busy ones) so IO kernels get co-scheduled; the
// IO streams additionally run at HIGH priority as a second guarantee.
// ---------------------------------------------------------------------------
__global__ void sgemm_layer_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int n, int iters) {
    const int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n * n; idx += stride) {
        const int row = idx / n;
        const int col = idx % n;
        for (int it = 0; it < iters; ++it) {
            float acc = 0.f;
            for (int k = 0; k < n; ++k)
                acc += A[row * n + k] * B[k * n + col];
            C[row * n + col] = acc;   // written every iter: never optimized away
        }
    }
}

int main(int argc, char** argv) {
    uint32_t n_layers      = 80;        // --layers: transformer layers (HY3 num_hidden_layers=80)
    uint64_t ctx_tokens    = 131072;    // --ctx-tokens: 128K context
    uint32_t chunk_tokens  = 256;       // --chunk-tokens: LMCache chunk size
    uint32_t hit_pct       = 90;        // --hit-pct: prefix-cache hit rate
    // Per-(chunk, layer) K (or V) tensor size.  HY3 (GQA): num_key_value_heads=8
    // x head_dim=128 x bf16 = 2 KiB/token/layer, so a 256-token LMCache chunk
    // is 512 KiB per K (or V) per layer -- this is the default.
    uint32_t tensor_kb     = 512;       // --tensor-kb: per (chunk,layer) K (or V)
    uint32_t n_requests    = 2;         // --requests: simulated requests
    uint64_t compute_us    = 0;         // --compute-us: per-layer compute time; 0 = auto
    uint32_t gemm_n        = 1024;      // --gemm-n: SGEMM matrix edge (fp32); keep 1 iter << target so calibration can subdivide
    uint32_t compute_sms   = 64;        // --compute-sms: SGEMM grid size in blocks (SM budget)
    uint64_t l1_mib        = 512;       // --l1-mib
    uint64_t l2_mib        = 2048;      // --l2-mib
    // Engine batch depth.  512 KiB tensors fan out to (512 KiB / min_mdts)
    // sub-IOs each -- 8 KiB covers mdts >= 64 KiB for 1024 tensors.
    uint32_t max_entries   = 8192;      // --max-entries
    bool     skip_create   = false;     // --skip-create: reuse existing kvlw_* files
    bool     recreate      = false;     // --recreate: wipe + re-create (needed after layout changes)
    bool     skip_delete   = false;     // --skip-delete
    bool     verify        = false;     // --verify: read back + check seeds afterwards
    tutti_examples::RegistryCliOptions ropt;

    for (int i = 1; i < argc; ) {
        if (tutti_examples::parse_registry_cli_arg(ropt, argc, argv, i)) { ++i; continue; }
        const char* a = argv[i];
        if      (std::strcmp(a, "--layers")     == 0 && i+1 < argc) { n_layers      = (uint32_t)std::strtoul(argv[++i], nullptr, 10); ++i; }
        else if (std::strcmp(a, "--ctx-tokens") == 0 && i+1 < argc) { ctx_tokens    = std::strtoull(argv[++i], nullptr, 10); ++i; }
        else if (std::strcmp(a, "--chunk-tokens")== 0 && i+1 < argc) { chunk_tokens = (uint32_t)std::strtoul(argv[++i], nullptr, 10); ++i; }
        else if (std::strcmp(a, "--hit-pct")    == 0 && i+1 < argc) { hit_pct       = (uint32_t)std::strtoul(argv[++i], nullptr, 10); ++i; }
        else if (std::strcmp(a, "--tensor-kb")  == 0 && i+1 < argc) { tensor_kb     = (uint32_t)std::strtoul(argv[++i], nullptr, 10); ++i; }
        else if (std::strcmp(a, "--requests")   == 0 && i+1 < argc) { n_requests    = (uint32_t)std::strtoul(argv[++i], nullptr, 10); ++i; }
        else if (std::strcmp(a, "--compute-us") == 0 && i+1 < argc) { compute_us    = std::strtoull(argv[++i], nullptr, 10); ++i; }
        else if (std::strcmp(a, "--gemm-n")     == 0 && i+1 < argc) { gemm_n        = (uint32_t)std::strtoul(argv[++i], nullptr, 10); ++i; }
        else if (std::strcmp(a, "--compute-sms")== 0 && i+1 < argc) { compute_sms   = (uint32_t)std::strtoul(argv[++i], nullptr, 10); ++i; }
        else if (std::strcmp(a, "--l1-mib")     == 0 && i+1 < argc) { l1_mib        = std::strtoull(argv[++i], nullptr, 10); ++i; }
        else if (std::strcmp(a, "--l2-mib")     == 0 && i+1 < argc) { l2_mib        = std::strtoull(argv[++i], nullptr, 10); ++i; }
        else if (std::strcmp(a, "--max-entries")== 0 && i+1 < argc) { max_entries   = (uint32_t)std::strtoul(argv[++i], nullptr, 10); ++i; }
        else if (std::strcmp(a, "--skip-create")== 0)                { skip_create   = true; ++i; }
        else if (std::strcmp(a, "--recreate")   == 0)                { recreate      = true; ++i; }
        else if (std::strcmp(a, "--skip-delete")== 0)                { skip_delete   = true; ++i; }
        else if (std::strcmp(a, "--verify")     == 0)                { verify        = true; ++i; }
        else { std::fprintf(stderr, "unknown arg: %s\n", a); return 1; }
    }
    if (!tutti_examples::validate_registry_cli(ropt, argv[0])) return 1;

    const uint64_t tensor_size    = (uint64_t)tensor_kb * 1024;
    const uint64_t n_chunks       = ctx_tokens / chunk_tokens;         // 128K/256 = 512
    const uint64_t n_hit          = n_chunks * hit_pct / 100;          // ~461
    const uint64_t n_miss         = n_chunks - n_hit;                  // ~51
    const uint64_t gpu_file_total = 2ull * n_layers * tensor_size;     // 2 shards
    if (n_chunks == 0 || n_hit == 0 || n_miss == 0)
        STEP_FAIL("bad geometry: chunks=%lu hit=%lu miss=%lu",
                  (unsigned long)n_chunks, (unsigned long)n_hit, (unsigned long)n_miss);

    CUDA_OK(cudaFree(0));
    CUDA_OK(cudaSetDevice(ropt.cuda_dev));
    STEP_OK("cudaSetDevice(%d)", ropt.cuda_dev);

    // ---- Coordinator bootstrap (SERVICE_CLIENT against the running daemon) ----
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
    cfg.num_user_queues_per_device = ropt.num_user_queues;  // --queues; daemon clamps to its max_per_client / kernel group cap
    cfg.max_entries_per_batch = max_entries;
    cfg.handle_l1_gpu_budget_bytes  = l1_mib * 1024 * 1024;
    cfg.handle_l2_host_budget_bytes = l2_mib * 1024 * 1024;

    tutti::Coordinator coord;
    if (!coord.bootstrap(cfg)) STEP_FAIL("Coordinator::bootstrap");
    STEP_OK("Coordinator bootstrap (%s, L1=%lu MiB, L2=%lu MiB)",
            ropt.service_mode ? "SERVICE_CLIENT" : "IN_PROCESS",
            (unsigned long)l1_mib, (unsigned long)l2_mib);

    auto coord_devs = coord.devices();
    if (coord_devs.size() < 2) STEP_FAIL("need >= 2 devices, got %zu", coord_devs.size());

    // ---- Phase A: create (or reuse) n_chunks GpuFiles "kvlw_<i>" ----
    std::vector<tutti::GpuFileId> chunk_ids;
    {
        auto names = coord.block()->list_gpu_file_names();
        std::size_t matching = 0;
        for (const auto& nm : names) if (nm.rfind("kvlw_", 0) == 0) ++matching;

        if ((skip_create || matching == n_chunks) && !recreate) {
            LOG_INFO("Phase A: reusing %zu existing kvlw_* GpuFiles", matching);
            std::vector<tutti::GpuFileSpec> specs;
            specs.reserve(matching);
            for (const auto& nm : names)
                if (nm.rfind("kvlw_", 0) == 0) specs.push_back({nm});
            auto gfs = coord.open_gpu_files_batch(
                specs.data(), (uint32_t)specs.size(), tutti::GPU_FILE_OPEN_EXISTING);
            for (auto* gf : gfs) if (gf) chunk_ids.push_back(gf->id);
        } else {
            if (matching > 0) {
                LOG_INFO("Phase A: %zu stale kvlw_* files != %lu chunks; removing first",
                         matching, (unsigned long)n_chunks);
                std::vector<tutti::GpuFileSpec> specs;
                for (const auto& nm : names)
                    if (nm.rfind("kvlw_", 0) == 0) specs.push_back({nm});
                auto gfs = coord.open_gpu_files_batch(
                    specs.data(), (uint32_t)specs.size(), tutti::GPU_FILE_OPEN_EXISTING);
                auto ok = std::make_unique<bool[]>(gfs.size());
                (void)coord.delete_gpu_files_batch(gfs.data(), (uint32_t)gfs.size(), ok.get());
            }
            auto t0 = std::chrono::steady_clock::now();
            std::vector<std::string> names_v(n_chunks);
            std::vector<tutti::GpuFileSpec> specs(n_chunks);
            for (uint64_t i = 0; i < n_chunks; ++i) {
                char nm[64]; std::snprintf(nm, sizeof(nm), "kvlw_%lu", (unsigned long)i);
                names_v[i] = nm;
                specs[i].name = names_v[i];
                specs[i].total_size = gpu_file_total;
                specs[i].tensor_shape[0] = 2;                 // K, V shards
                specs[i].tensor_shape[1] = n_layers;
                specs[i].tensor_shape[2] = (uint32_t)tensor_size;
                // Alternate K/V shard pairs across ALL attached devices
                // ({dev0,dev1}, {dev2,dev3}, ... round-robin): every batch
                // then mixes chunks living on every device, so a single
                // nvme_batch_xfer_kernel drives all N NVMes concurrently
                // (aggregate bandwidth), with no adapter changes needed.
                const size_t ndev = coord_devs.size();
                specs[i].shard_placement = {coord_devs[(2 * i)     % ndev],
                                            coord_devs[(2 * i + 1) % ndev]};
            }
            auto gfs = coord.open_gpu_files_batch(
                specs.data(), (uint32_t)specs.size(),
                tutti::GPU_FILE_OPEN_CREATE | tutti::GPU_FILE_OPEN_NO_PERSIST);
            for (auto* gf : gfs) {
                if (gf == nullptr) STEP_FAIL("Phase A: GpuFile create failed");
                chunk_ids.push_back(gf->id);
            }
            for (auto* d : coord_devs) coord.storage()->flush_metadata(d);
            coord.block()->flush_metadata();
            STEP_OK("Phase A: created %lu GpuFiles (%.1f GB total) in %.2fs",
                    (unsigned long)n_chunks,
                    (double)(n_chunks * gpu_file_total) / (1024.0*1024.0*1024.0),
                    seconds_since(t0));
        }
    }
    if (chunk_ids.size() != n_chunks)
        STEP_FAIL("Phase A: chunk count mismatch: %zu != %lu",
                  chunk_ids.size(), (unsigned long)n_chunks);

    // hit chunks = first n_hit ids, miss chunks = the rest.
    std::vector<tutti::GpuFileId> hit_fids(chunk_ids.begin(), chunk_ids.begin() + n_hit);
    std::vector<tutti::GpuFileId> miss_fids(chunk_ids.begin() + n_hit, chunk_ids.end());

    // ---- Phase B: allocate + register per-chunk K/V tensors (full 128K
    //      context resident on GPU, reused across layers -- mirrors vLLM's
    //      KV cache residency). ----
    std::vector<tutti::MemoryRegion*> hit_k(n_hit), hit_v(n_hit);
    std::vector<tutti::MemoryRegion*> miss_k(n_miss), miss_v(n_miss);
    auto alloc_reg = [&](std::vector<tutti::MemoryRegion*>& k,
                         std::vector<tutti::MemoryRegion*>& v, uint64_t n) {
        for (uint64_t b = 0; b < n; ++b) {
            k[b] = coord.allocate_device(tensor_size, tutti::MemoryKind::DEVICE, ropt.cuda_dev);
            v[b] = coord.allocate_device(tensor_size, tutti::MemoryKind::DEVICE, ropt.cuda_dev);
            if (!k[b] || !v[b]) STEP_FAIL("Phase B: allocate_device #%lu", (unsigned long)b);
            tutti::TensorRegistrationSpec ds{};
            ds.size = tensor_size; ds.granularity = tensor_size;
            ds.ptr = k[b]->device_ptr;
            if (coord.register_tensor(ds) == nullptr)
                STEP_FAIL("Phase B: register_tensor K #%lu (BAR1 window?)", (unsigned long)b);
            ds.ptr = v[b]->device_ptr;
            if (coord.register_tensor(ds) == nullptr)
                STEP_FAIL("Phase B: register_tensor V #%lu (BAR1 window?)", (unsigned long)b);
        }
    };
    alloc_reg(hit_k, hit_v, n_hit);
    alloc_reg(miss_k, miss_v, n_miss);
    STEP_OK("Phase B: %lu hit + %lu miss chunks resident (%.1f MiB GPU)",
            (unsigned long)n_hit, (unsigned long)n_miss,
            (double)(2 * n_chunks * tensor_size) / (1024.0*1024.0));

    // IO streams run at HIGH priority: even when the compute kernel
    // still has queued blocks, nvme_batch_xfer_kernel blocks win the
    // scheduler first.  (Lower numeric value == higher priority.)
    int prio_lo = 0, prio_hi = 0;
    CUDA_OK(cudaDeviceGetStreamPriorityRange(&prio_lo, &prio_hi));
    cudaStream_t s_read, s_compute, s_write;
    CUDA_OK(cudaStreamCreateWithPriority(&s_read, cudaStreamDefault, prio_hi));
    CUDA_OK(cudaStreamCreateWithPriority(&s_compute, cudaStreamDefault, prio_lo));
    CUDA_OK(cudaStreamCreateWithPriority(&s_write, cudaStreamDefault, prio_hi));

    tutti::KvCacheIoAdapter kv(coord, tensor_size, /*use_mla=*/false);

    // ---- Phase C: pre-write HIT chunks to SSD (setup; these are the
    //      "previously cached" 90%).  Miss chunks are created empty and
    //      get written during the simulated pass. ----
    {
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t L = 0; L < n_layers; ++L) {
            for (uint64_t b = 0; b < n_hit; ++b) {
                uint8_t seed = (uint8_t)((b * n_layers + L) & 0xFF);
                CUDA_OK(cudaMemsetAsync(hit_k[b]->device_ptr, seed, tensor_size, s_write));
                CUDA_OK(cudaMemsetAsync(hit_v[b]->device_ptr, seed ^ 0xA5, tensor_size, s_write));
            }
            CUDA_OK(cudaStreamSynchronize(s_write));
            if (!kv.batched_write((int)L, hit_k, hit_v, hit_fids, s_write))
                STEP_FAIL("Phase C: prewrite batched_write L=%u", L);
        }
        for (auto fid : hit_fids) coord.sync_file(fid, s_write);
        CUDA_OK(cudaStreamSynchronize(s_write));
        STEP_OK("Phase C: pre-wrote %lu hit chunks x %u layers (%.2f GB) in %.2fs",
                (unsigned long)n_hit, (unsigned)n_layers,
                (double)(n_hit * 2 * n_layers * tensor_size) / (1024.0*1024.0*1024.0),
                seconds_since(t0));
    }

    // ---- Phase D: SGEMM buffers + per-layer compute calibration ----
    // A real SGEMM stands in for one layer's math; the iteration count
    // is calibrated so one layer's compute ~= one layer's IO time.
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    const size_t gemm_bytes = (size_t)gemm_n * gemm_n * sizeof(float);
    CUDA_OK(cudaMalloc(&d_A, gemm_bytes));
    CUDA_OK(cudaMalloc(&d_B, gemm_bytes));
    CUDA_OK(cudaMalloc(&d_C, gemm_bytes));
    CUDA_OK(cudaMemset(d_A, 0x3f, gemm_bytes));   // arbitrary non-zero patterns
    CUDA_OK(cudaMemset(d_B, 0x2b, gemm_bytes));
    // Small 1-D grid: compute_sms blocks cover the matrix via the
    // kernel's grid-stride loop, leaving the remaining SMs free for
    // co-scheduled IO kernels (see the kernel's comment).
    const dim3 gemm_block(256);
    const dim3 gemm_grid(compute_sms);

    // Warm up (JIT/caches), then time ONE SGEMM iteration.
    cudaEvent_t ew0, ew1;
    CUDA_OK(cudaEventCreate(&ew0));
    CUDA_OK(cudaEventCreate(&ew1));
    sgemm_layer_kernel<<<gemm_grid, gemm_block, 0, s_compute>>>(d_A, d_B, d_C, (int)gemm_n, 1);
    CUDA_OK(cudaStreamSynchronize(s_compute));
    CUDA_OK(cudaEventRecord(ew0, s_compute));
    sgemm_layer_kernel<<<gemm_grid, gemm_block, 0, s_compute>>>(d_A, d_B, d_C, (int)gemm_n, 1);
    CUDA_OK(cudaEventRecord(ew1, s_compute));
    CUDA_OK(cudaEventSynchronize(ew1));
    float gemm_ms = 0.f;
    CUDA_OK(cudaEventElapsedTime(&gemm_ms, ew0, ew1));
    CUDA_OK(cudaEventDestroy(ew0));
    CUDA_OK(cudaEventDestroy(ew1));
    if (gemm_ms <= 0.f) STEP_FAIL("Phase D: SGEMM timing failed");

    if (compute_us == 0) {
        auto tr = std::chrono::steady_clock::now();
        if (!kv.batched_read(0, hit_k, hit_v, hit_fids, s_read))
            STEP_FAIL("Phase D: calib read");
        double t_read = seconds_since(tr);
        auto tw = std::chrono::steady_clock::now();
        if (!kv.batched_write(0, miss_k, miss_v, miss_fids, s_write))
            STEP_FAIL("Phase D: calib write");
        double t_write = seconds_since(tw);
        compute_us = (uint64_t)((t_read + t_write) * 1e6);
        STEP_OK("Phase D: auto compute_us=%lu us (read %.3f ms + write %.3f ms per layer)",
                (unsigned long)compute_us, t_read * 1e3, t_write * 1e3);
    }
    const uint32_t compute_iters =
        (uint32_t)std::max(1.0, (double)compute_us / ((double)gemm_ms * 1000.0) + 0.5);
    LOG_INFO("pipeline: layers=%u chunks=%lu (hit=%lu miss=%lu) compute=%lu us/layer "
             "= %u SGEMM iters (n=%u, %.3f ms/iter)",
             (unsigned)n_layers, (unsigned long)n_chunks,
             (unsigned long)n_hit, (unsigned long)n_miss,
             (unsigned long)compute_us, (unsigned)compute_iters,
             (unsigned)gemm_n, (double)gemm_ms);

    // Per-layer dependency events (no timing needed).
    std::vector<cudaEvent_t> ev_read(n_layers), ev_comp(n_layers);
    for (auto& e : ev_read) CUDA_OK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    for (auto& e : ev_comp) CUDA_OK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));

    // IO interface timing: GPU-side events wrapping each batched_read /
    // batched_write.  Measured on the IO stream itself, so a write's
    // time excludes any wait for its compute dependency (the t0 record
    // lands AFTER the stream's waitEvent).  These two carry timing,
    // unlike the DisableTiming dependency events above.
    cudaEvent_t ev_t0, ev_t1;
    CUDA_OK(cudaEventCreate(&ev_t0));
    CUDA_OK(cudaEventCreate(&ev_t1));
    auto time_io = [&](cudaStream_t s, auto&& fn) -> float {
        CUDA_OK(cudaEventRecord(ev_t0, s));
        fn();
        CUDA_OK(cudaEventRecord(ev_t1, s));
        CUDA_OK(cudaEventSynchronize(ev_t1));
        float ms = 0.f;
        CUDA_OK(cudaEventElapsedTime(&ms, ev_t0, ev_t1));
        return ms;
    };

    // ---- Phase E: the simulated serving pass (nsys records THIS window) ----
    const double read_layer_gb  = (double)(n_hit  * 2 * tensor_size) / 1e9;
    const double write_layer_gb = (double)(n_miss * 2 * tensor_size) / 1e9;
    double sim_wall = 0.0, sim_read_ms = 0.0, sim_write_ms = 0.0;
    CUDA_OK(cudaProfilerStart());
    for (uint32_t req = 0; req < n_requests; ++req) {
        nvtxRangePushA("request");
        auto t_req0 = std::chrono::steady_clock::now();
        double req_read_ms = 0.0, req_write_ms = 0.0;
        float last_read_ms = 0.f, last_write_ms = 0.f;

        // Prefetch layer 0 of the hit set before the loop starts.
        nvtxRangePushA("read L0 (prefetch)");
        last_read_ms = time_io(s_read, [&] {
            if (!kv.batched_read(0, hit_k, hit_v, hit_fids, s_read))
                STEP_FAIL("Phase E: read L0");
        });
        req_read_ms += last_read_ms;
        nvtxRangePop();
        CUDA_OK(cudaEventRecord(ev_read[0], s_read));

        for (uint32_t L = 0; L < n_layers; ++L) {
            char span[64];

            // (1) compute(L): waits for read(L) to have landed; async
            //     launch -- the host does NOT wait for it, which is
            //     what lets it overlap the IO below.
            std::snprintf(span, sizeof(span), "compute L%u", L);
            CUDA_OK(cudaStreamWaitEvent(s_compute, ev_read[L], 0));
            nvtxRangePushA(span);
            sgemm_layer_kernel<<<gemm_grid, gemm_block, 0, s_compute>>>(
                d_A, d_B, d_C, (int)gemm_n, (int)compute_iters);
            CUDA_OK(cudaGetLastError());
            nvtxRangePop();
            CUDA_OK(cudaEventRecord(ev_comp[L], s_compute));

            // (2) write(L-1): store the miss KV "produced" by
            //     compute(L-1).  Blocks the host; compute(L) runs
            //     concurrently on the GPU.  IO time measured GPU-side,
            //     excluding the wait on ev_comp (see time_io).
            if (L >= 1) {
                std::snprintf(span, sizeof(span), "write L%u (miss)", L - 1);
                nvtxRangePushA(span);
                CUDA_OK(cudaStreamWaitEvent(s_write, ev_comp[L - 1], 0));
                last_write_ms = time_io(s_write, [&] {
                    if (!kv.batched_write((int)(L - 1), miss_k, miss_v, miss_fids, s_write))
                        STEP_FAIL("Phase E: write L=%u", L - 1);
                });
                req_write_ms += last_write_ms;
                nvtxRangePop();
            }

            // (3) read(L+1): prefetch the next layer's hit KV.
            //     Blocks the host; compute(L) runs concurrently.
            if (L + 1 < n_layers) {
                std::snprintf(span, sizeof(span), "read L%u (hit)", L + 1);
                nvtxRangePushA(span);
                last_read_ms = time_io(s_read, [&] {
                    if (!kv.batched_read((int)(L + 1), hit_k, hit_v, hit_fids, s_read))
                        STEP_FAIL("Phase E: read L=%u", L + 1);
                });
                req_read_ms += last_read_ms;
                nvtxRangePop();
                CUDA_OK(cudaEventRecord(ev_read[L + 1], s_read));
            }

            // Per-10-layer interface-speed report.
            if (L % 10 == 9 || L + 1 == n_layers) {
                LOG_INFO("req%u L%-3u IO speed: read %.1f MB / %.2f ms = %.2f GB/s | "
                         "write %.1f MB / %.2f ms = %.2f GB/s",
                         req + 1, L,
                         read_layer_gb * 1e3, (double)last_read_ms,
                         last_read_ms > 0 ? read_layer_gb / ((double)last_read_ms / 1e3) : 0.0,
                         write_layer_gb * 1e3, (double)last_write_ms,
                         last_write_ms > 0 ? write_layer_gb / ((double)last_write_ms / 1e3) : 0.0);
            }
        }

        // Drain: last layer's miss write.
        nvtxRangePushA("write last (miss)");
        CUDA_OK(cudaStreamWaitEvent(s_write, ev_comp[n_layers - 1], 0));
        last_write_ms = time_io(s_write, [&] {
            if (!kv.batched_write((int)(n_layers - 1), miss_k, miss_v, miss_fids, s_write))
                STEP_FAIL("Phase E: write last layer");
        });
        req_write_ms += last_write_ms;
        nvtxRangePop();
        for (auto fid : miss_fids) coord.sync_file(fid, s_write);
        CUDA_OK(cudaStreamSynchronize(s_compute));
        CUDA_OK(cudaStreamSynchronize(s_read));
        CUDA_OK(cudaStreamSynchronize(s_write));
        nvtxRangePop();  // request

        double req_wall = seconds_since(t_req0);
        sim_wall += req_wall;
        sim_read_ms += req_read_ms;
        sim_write_ms += req_write_ms;
        const double req_read_gb  = read_layer_gb  * n_layers;
        const double req_write_gb = write_layer_gb * n_layers;
        // Serial baseline: what this request would cost with NO
        // overlap (all IO time + all compute time, end to end).
        const double serial_s = (req_read_ms + req_write_ms) / 1e3
                              + (double)n_layers * (double)compute_us / 1e6;
        STEP_OK("Phase E: request %u done in %.3fs (serial baseline %.3fs, overlap saving %.0f%%)",
                req + 1, req_wall, serial_s, 100.0 * (1.0 - req_wall / serial_s));
        STEP_OK("Phase E:   READ  interface: %.2f GB in %.1f ms IO-time = %.2f GB/s",
                req_read_gb, req_read_ms, req_read_gb / (req_read_ms / 1e3));
        STEP_OK("Phase E:   WRITE interface: %.2f GB in %.1f ms IO-time = %.2f GB/s",
                req_write_gb, req_write_ms, req_write_gb / (req_write_ms / 1e3));
    }
    CUDA_OK(cudaProfilerStop());

    const double total_read_gb  = read_layer_gb  * n_layers * n_requests;
    const double total_write_gb = write_layer_gb * n_layers * n_requests;
    STEP_OK("SIMULATION TOTAL: %u requests, wall=%.3fs | READ interface %.2f GB = %.2f GB/s "
            "| WRITE interface %.2f GB = %.2f GB/s",
            (unsigned)n_requests, sim_wall,
            total_read_gb,  total_read_gb  / (sim_read_ms  / 1e3),
            total_write_gb, total_write_gb / (sim_write_ms / 1e3));

    // ---- Phase F (optional): verify a sample of seeds ----
    if (verify) {
        void* h_buf = nullptr;
        CUDA_OK(cudaMallocHost(&h_buf, tensor_size));
        uint64_t mismatch = 0;
        for (uint64_t b = 0; b < n_hit; b += std::max<uint64_t>(1, n_hit / 16)) {
            uint32_t L = (uint32_t)(b % n_layers);
            uint8_t expect = (uint8_t)((b * n_layers + L) & 0xFF);
            CUDA_OK(cudaMemsetAsync(hit_k[b]->device_ptr, 0, tensor_size, s_read));
            std::vector<tutti::MemoryRegion*> one_k{hit_k[b]}, one_v{hit_v[b]};
            std::vector<tutti::GpuFileId> one_f{hit_fids[b]};
            if (!kv.batched_read((int)L, one_k, one_v, one_f, s_read))
                STEP_FAIL("Phase F: verify read b=%lu", (unsigned long)b);
            CUDA_OK(cudaMemcpyAsync(h_buf, hit_k[b]->device_ptr, tensor_size,
                                    cudaMemcpyDeviceToHost, s_read));
            CUDA_OK(cudaStreamSynchronize(s_read));
            if (((uint8_t*)h_buf)[0] != expect) ++mismatch;
        }
        CUDA_OK(cudaFreeHost(h_buf));
        if (mismatch != 0) STEP_FAIL("Phase F: %lu seed mismatches", (unsigned long)mismatch);
        STEP_OK("Phase F: sampled hit-chunk seeds verified");
    }

    // ---- Cleanup ----
    CUDA_OK(cudaFree(d_A));
    CUDA_OK(cudaFree(d_B));
    CUDA_OK(cudaFree(d_C));
    CUDA_OK(cudaEventDestroy(ev_t0));
    CUDA_OK(cudaEventDestroy(ev_t1));
    for (auto& e : ev_read) CUDA_OK(cudaEventDestroy(e));
    for (auto& e : ev_comp) CUDA_OK(cudaEventDestroy(e));
    for (auto* r : hit_k)  if (r) coord.free(r);
    for (auto* r : hit_v)  if (r) coord.free(r);
    for (auto* r : miss_k) if (r) coord.free(r);
    for (auto* r : miss_v) if (r) coord.free(r);
    CUDA_OK(cudaStreamDestroy(s_read));
    CUDA_OK(cudaStreamDestroy(s_compute));
    CUDA_OK(cudaStreamDestroy(s_write));

    if (skip_delete) {
        STEP_OK("cleanup: --skip-delete, %zu kvlw_* files left in place",
                coord.block()->list_gpu_file_names().size());
    } else {
        auto names = coord.block()->list_gpu_file_names();
        std::vector<tutti::GpuFileSpec> specs;
        for (const auto& nm : names)
            if (nm.rfind("kvlw_", 0) == 0) specs.push_back({nm});
        auto gfs = coord.open_gpu_files_batch(
            specs.data(), (uint32_t)specs.size(), tutti::GPU_FILE_OPEN_EXISTING);
        auto ok = std::make_unique<bool[]>(gfs.size());
        (void)coord.delete_gpu_files_batch(gfs.data(), (uint32_t)gfs.size(), ok.get());
        for (auto* d : coord_devs) coord.storage()->flush_metadata(d);
        coord.block()->flush_metadata();
        STEP_OK("cleanup: deleted %zu kvlw_* files", gfs.size());
    }

    coord.shutdown();
    std::fprintf(stderr, "\n=== kv_cache_layerwise_overlap: PASSED ===\n");
    return 0;
}
