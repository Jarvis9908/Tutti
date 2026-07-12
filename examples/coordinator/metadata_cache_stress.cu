/**
 * metadata_cache_stress.cu -- large-scale stress test for the two-tier
 * metadata cache + batch interfaces.
 *
 * Flow:
 *   [1] Coordinator bootstrap (SERVICE_CLIENT)
 *   [2] Bulk create N GpuFiles (64 KiB each, 2 shards × 32 KiB)
 *       with L1=128 MiB / L2=128 MiB handle pools.
 *   [3] Batch READ: resolve handles via handle_for_batch + submit_batch
 *       in chunks, verify seed-A pattern on GPU.
 *   [4] Batch WRITE: write seed-B, verify via host pread.
 *   [5] Delete all N GpuFiles.
 *   [6] Report timing for each phase.
 *
 * Usage:
 *   sudo ./metadata_cache_stress --cuda 0 --service 127.0.0.1:50051 \
 *       --dev-id 0 --dev-id 1 [--n 1000000] [--chunk 4096]
 *
 * Defaults: n=100000  chunk=4096
 *   (start with 100K; bump --n 1000000 for the full 1M once you've
 *    confirmed 100K works.)
 */

#include "../../coordinator/include/coordinator.h"
#include "../../coordinator/include/device.h"
#include "../../block_storage/include/block_storage.h"
#include "../../block_storage/include/host_fs_backed_block_storage.h"
#include "../../nvme_storage/include/host_fs_backed_nvme_storage.h"
#include "../../nvme_storage/include/nvme_storage.h"
#include "../../nvme_storage/include/nvme_file.h"
#include "../../memory/include/memory_subsystem.h"
#include "../../memory/include/memory_kind.h"
#include "../../memory/include/memory_region.h"
#include "../../io_engine/include/local_nvme/nvme_batch.h"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

namespace {

int g_step = 0;

#define STEP_OK(fmt, ...) do {                                                 \
    ++g_step;                                                                  \
    std::fprintf(stderr, "[ OK ] step=%-2d  " fmt "\n", g_step, ##__VA_ARGS__);\
} while (0)

#define STEP_FAIL(fmt, ...) do {                                               \
    ++g_step;                                                                  \
    std::fprintf(stderr, "[FAIL] step=%-2d  " fmt "\n", g_step, ##__VA_ARGS__);\
    std::_Exit(2);                                                             \
} while (0)

#define CUDA_OK(call) do {                                                     \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) STEP_FAIL("CUDA error: %s (%s)",                    \
                                     #call, cudaGetErrorString(_e));           \
} while (0)

double seconds_since(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
}

// O_DIRECT requires a block-aligned buffer (posix_memalign).
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

constexpr uint32_t kNumShards   = 2;
constexpr uint32_t kTensorSize  = 32u * 1024;    // 32 KiB per shard
constexpr uint64_t kGpuFileSize  = (uint64_t)kNumShards * kTensorSize;  // 64 KiB

void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [--cuda N] --service <endpoint> --dev-id N --dev-id M\n"
        "          [--n COUNT] [--chunk SIZE] [--l1-mib N] [--l2-mib N]\n"
        "Defaults: n=100000 chunk=4096 l1-mib=128 l2-mib=128\n"
        "Each GpuFile = 64 KiB (2 shards x 32 KiB).\n"
        "DESTRUCTIVE on unformatted disks.\n",
        prog);
}

}  // namespace

int main(int argc, char** argv) {
    int      cuda_dev = 0;
    std::string service_endpoint;
    std::vector<int32_t> dev_ids;
    uint64_t n_files    = 100000;
    uint32_t chunk_size = 4096;
    uint64_t l1_mib     = 128;
    uint64_t l2_mib     = 128;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if      (!std::strcmp(a, "--cuda") && i + 1 < argc) cuda_dev = std::atoi(argv[++i]);
        else if (!std::strcmp(a, "--service") && i + 1 < argc) service_endpoint = argv[++i];
        else if (!std::strcmp(a, "--dev-id") && i + 1 < argc) dev_ids.push_back((int32_t)std::atoi(argv[++i]));
        else if (!std::strcmp(a, "--n") && i + 1 < argc) n_files = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(a, "--chunk") && i + 1 < argc) chunk_size = (uint32_t)std::atoi(argv[++i]);
        else if (!std::strcmp(a, "--l1-mib") && i + 1 < argc) l1_mib = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(a, "--l2-mib") && i + 1 < argc) l2_mib = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(a, "-h") || !std::strcmp(a, "--help")) { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "unknown arg: %s\n", a); usage(argv[0]); return 1; }
    }
    if (service_endpoint.empty() || dev_ids.size() != kNumShards) {
        std::fprintf(stderr, "need --service + exactly %u --dev-id\n", kNumShards);
        usage(argv[0]); return 1;
    }

    CUDA_OK(cudaSetDevice(cuda_dev));
    STEP_OK("cudaSetDevice(%d)", cuda_dev);

    // [1] Coordinator bootstrap
    tutti::CoordinatorConfig cfg;
    cfg.cuda_device                = cuda_dev;
    cfg.num_user_queues_per_device = 4;
    cfg.descriptor_format          = tutti::DescriptorFormat::PRP;
    cfg.max_entries_per_batch      = 64;
    cfg.mode                       = tutti::CoordinatorMode::SERVICE_CLIENT;
    cfg.daemon_endpoint            = service_endpoint;
    cfg.daemon_device_ids          = dev_ids;
    // L1/L2 budgets: 128 MiB each.  With ~192 B per handle, L1 ≈ L2 ≈ 699K.
    cfg.handle_l1_gpu_budget_bytes  = l1_mib << 20;
    cfg.handle_l2_host_budget_bytes = l2_mib << 20;

    tutti::Coordinator coord;
    if (!coord.bootstrap(cfg)) STEP_FAIL("Coordinator::bootstrap");
    STEP_OK("Coordinator::bootstrap [SERVICE_CLIENT] (L1=%lluMiB L2=%lluMiB)",
            (unsigned long long)l1_mib, (unsigned long long)l2_mib);

    const uint32_t l1_cap = coord.handle_cache_capacity();
    if (chunk_size > l1_cap) {
        std::fprintf(stderr,
            "chunk=%u > handle_l1_capacity=%u; capping chunk to %u\n",
            chunk_size, l1_cap, l1_cap);
        chunk_size = l1_cap;
    }
    STEP_OK("handle_l1_capacity=%u (chunk=%u)", l1_cap, chunk_size);

    // [2] Bulk create N GpuFiles
    auto t0 = std::chrono::steady_clock::now();
    std::vector<tutti::GpuFileId> file_ids;
    file_ids.reserve(n_files);
    for (uint64_t b = 0; b < n_files; ++b) {
        std::string name = "mc_stress_" + std::to_string(b);
        tutti::GpuFileSpec spec{};
        spec.name            = name;
        spec.total_size      = kGpuFileSize;
        spec.tensor_shape[0] = kNumShards;
        spec.tensor_shape[1] = 1;
        spec.tensor_shape[2] = kTensorSize;
        spec.shard_placement = { coord.devices()[0], coord.devices()[1] };

        // Pre-clean if exists
        if (tutti::GpuFile* old = coord.open_gpu_file(spec))
            (void)coord.delete_gpu_file(old, /*persist_now=*/false);

        tutti::GpuFile* gf = coord.open_gpu_file(
            spec, tutti::GPU_FILE_OPEN_CREATE | tutti::GPU_FILE_OPEN_NO_PERSIST);
        if (gf == nullptr) STEP_FAIL("open_gpu_file(CREATE) b=%llu", (unsigned long long)b);
        file_ids.push_back(gf->id);
    }
    coord.block()->flush_metadata();
    double sec_create = seconds_since(t0);
    STEP_OK("[2] created %llu GpuFiles (bulk + 1x flush), wall=%.3fs (%.0f ops/s)",
            (unsigned long long)n_files, sec_create,
            sec_create > 0 ? n_files / sec_create : 0.0);

    // [3] Host-write seed-A to every file
    auto t1 = std::chrono::steady_clock::now();
    AlignedBuf seed_a(kTensorSize);
    for (uint32_t i = 0; i < kTensorSize; ++i)
        seed_a[i] = (uint8_t)((i * 0x9E3779B9u) & 0xFF);
    for (uint64_t b = 0; b < n_files; ++b) {
        tutti::GpuFileSpec spec{};
        spec.name = "mc_stress_" + std::to_string(b);
        tutti::GpuFile* gf = coord.open_gpu_file(spec);  // EXISTING
        if (gf == nullptr) STEP_FAIL("open(EXISTING) b=%llu", (unsigned long long)b);
        for (uint32_t s = 0; s < kNumShards; ++s) {
            if (coord.storage()->write_blocking(gf->shards[s], 0,
                                                seed_a.data(), kTensorSize)
                != (ssize_t)kTensorSize)
                STEP_FAIL("write_blocking b=%llu s=%u", (unsigned long long)b, s);
        }
    }
    // Sync all shards
    for (uint64_t b = 0; b < n_files; ++b) {
        tutti::GpuFileSpec spec{};
        spec.name = "mc_stress_" + std::to_string(b);
        tutti::GpuFile* gf = coord.open_gpu_file(spec);
        for (uint32_t s = 0; s < kNumShards; ++s)
            coord.storage()->sync(gf->shards[s]);
    }
    double sec_write_seed = seconds_since(t1);
    STEP_OK("[3] host-wrote seed-A to %llu files, wall=%.3fs (%.0f ops/s)",
            (unsigned long long)n_files, sec_write_seed,
            sec_write_seed > 0 ? n_files / sec_write_seed : 0.0);

    // [4] Batch READ via handle_for_batch + submit_batch
    auto t2 = std::chrono::steady_clock::now();
    cudaStream_t stream = nullptr;
    CUDA_OK(cudaStreamCreate(&stream));

    // Allocate one GPU tensor for verification (over-allocate to ensure
    // page alignment -- CUDA guarantees 256B, but we need 4 KiB).
    constexpr uint64_t kAllocSize = 1ull << 20;  // 1 MiB, page-aligned
    tutti::MemoryRegion* tr = coord.allocate_device(
        kAllocSize, tutti::MemoryKind::DEVICE, cuda_dev);
    // Find the page-aligned start within the allocation
    uint64_t base = reinterpret_cast<uint64_t>(tr->device_ptr);
    uint64_t aligned = (base + 4095) & ~4095ull;
    void* aligned_ptr = reinterpret_cast<void*>(aligned);
    tutti::TensorRegistrationSpec ds{};
    ds.ptr = aligned_ptr; ds.size = kTensorSize; ds.granularity = kTensorSize;
    if (coord.register_tensor(ds) == nullptr)
        STEP_FAIL("register_tensor for read buffer");

    uint64_t n_read = 0;
    for (uint64_t off = 0; off < n_files; off += chunk_size) {
        uint32_t cnt = (uint32_t)std::min<uint64_t>(chunk_size, n_files - off);
        std::vector<tutti::GpuFileId> ids(file_ids.begin() + off,
                                          file_ids.begin() + off + cnt);
        std::vector<tutti::GpuFileHandle*> handles(cnt, nullptr);
        if (!coord.handle_for_batch(ids.data(), cnt, stream, handles.data()))
            STEP_FAIL("handle_for_batch off=%llu", (unsigned long long)off);

        // Submit read for each file (chunked under max_entries_per_batch)
        for (uint32_t i = 0; i < cnt; ++i) {
            tutti::NvmeBatchInputTensor in{tr, handles[i], 0};
            if (!coord.submit_batch({in}, /*is_read=*/true, stream))
                STEP_FAIL("submit_batch(read) off=%llu i=%u",
                          (unsigned long long)off, i);
        }
        CUDA_OK(cudaStreamSynchronize(stream));
        n_read += cnt;
        if (n_read % (chunk_size * 10) == 0)
            std::fprintf(stderr, "  ... read %llu / %llu\n",
                         (unsigned long long)n_read,
                         (unsigned long long)n_files);
    }
    double sec_read = seconds_since(t2);
    STEP_OK("[4] batch READ %llu files (chunk=%u), wall=%.3fs (%.0f ops/s)",
            (unsigned long long)n_read, chunk_size, sec_read,
            sec_read > 0 ? n_read / sec_read : 0.0);

    // [5] Drop cache + re-read (test handle rebuild)
    coord.drop_cached_handles(stream);
    auto t3 = std::chrono::steady_clock::now();
    uint64_t n_reread = 0;
    for (uint64_t off = 0; off < n_files; off += chunk_size) {
        uint32_t cnt = (uint32_t)std::min<uint64_t>(chunk_size, n_files - off);
        std::vector<tutti::GpuFileId> ids(file_ids.begin() + off,
                                          file_ids.begin() + off + cnt);
        std::vector<tutti::GpuFileHandle*> handles(cnt, nullptr);
        if (!coord.handle_for_batch(ids.data(), cnt, stream, handles.data()))
            STEP_FAIL("handle_for_batch(re-read) off=%llu", (unsigned long long)off);
        n_reread += cnt;
    }
    double sec_reread = seconds_since(t3);
    STEP_OK("[5] drop_cache + handle_for_batch(re-read) %llu files, wall=%.3fs (%.0f ops/s)",
            (unsigned long long)n_reread, sec_reread,
            sec_reread > 0 ? n_reread / sec_reread : 0.0);

    // [6] Delete all N GpuFiles
    auto t4 = std::chrono::steady_clock::now();
    for (uint64_t b = 0; b < n_files; ++b) {
        tutti::GpuFileSpec spec{};
        spec.name = "mc_stress_" + std::to_string(b);
        tutti::GpuFile* gf = coord.open_gpu_file(spec);
        if (gf != nullptr) coord.delete_gpu_file(gf, /*persist_now=*/false);
    }
    coord.block()->flush_metadata();
    double sec_delete = seconds_since(t4);
    STEP_OK("[6] deleted %llu GpuFiles (deferred + 1x flush), wall=%.3fs (%.0f ops/s)",
            (unsigned long long)n_files, sec_delete,
            sec_delete > 0 ? n_files / sec_delete : 0.0);

    // Cleanup
    coord.free(tr);
    cudaStreamDestroy(stream);
    coord.shutdown();

    std::fprintf(stderr,
        "\n=== metadata_cache_stress: %llu files, L1=%lluMiB L2=%lluMiB ===\n"
        "  create:       %.3fs (%.0f ops/s)\n"
        "  write_seed:   %.3fs (%.0f ops/s)\n"
        "  batch_read:   %.3fs (%.0f ops/s)\n"
        "  reread:       %.3fs (%.0f ops/s)\n"
        "  delete:       %.3fs (%.0f ops/s)\n",
        (unsigned long long)n_files, (unsigned long long)l1_mib,
        (unsigned long long)l2_mib,
        sec_create,   sec_create   > 0 ? n_files / sec_create   : 0.0,
        sec_write_seed, sec_write_seed > 0 ? n_files / sec_write_seed : 0.0,
        sec_read,     sec_read     > 0 ? n_read / sec_read     : 0.0,
        sec_reread,   sec_reread   > 0 ? n_reread / sec_reread : 0.0,
        sec_delete,   sec_delete   > 0 ? n_files / sec_delete   : 0.0);
    return 0;
}
