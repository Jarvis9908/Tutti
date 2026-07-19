/**
 * e2e_smoke.cu -- end-to-end smoke driven entirely through the
 * Coordinator.
 *
 * Proves that a single Coordinator::bootstrap() brings up
 * the whole stack (registry -> nvme_storage -> block_storage ->
 * memory -> io_engine), and the high-level passthroughs
 * (open_gpu_file / acquire_device_handle / allocate_device /
 * register_tensor / submit_batch) drive a byte-verified roundtrip in
 * both directions.
 *
 * Layout (same shape as io_engine_smoke):
 *   GpuFile  : num_shards=2, layers=4, tensor_size=128 KiB (1 MiB total)
 *   tensor   : 1 MiB device region, granularity=128 KiB -> 8 sub-slices
 *
 * Flow:
 *   READ  : host pwrite seed-A to shards -> coord.submit_batch(read)
 *           -> verify device tensor == seed-A
 *   WRITE : seed-B -> device tensor -> coord.submit_batch(write)
 *           -> drop page cache -> host pread -> verify shards == seed-B
 *
 * Bring-up mode: defaults to IN_PROCESS (this process owns the
 * NVMe).  Pass --service <endpoint> to drive the SAME stack through a
 * running tutti_daemon over the SERVICE_CLIENT path -- the daemon owns
 * chrdev/bind, this process attaches as a libnvm client and builds its
 * own user queue group.
 *
 * Test contract: DESTRUCTIVE on unformatted disks; exactly 2 shards.
 *   IN_PROCESS     : sole NVMe owner (no tutti_daemon running), 2 BDFs.
 *   SERVICE_CLIENT : tutti_daemon already up + owning the NVMe; 2
 *                    daemon-local device ids (default 0 1).
 *
 * Usage:
 *   sudo ./e2e_smoke --cuda 0 --cap 32 0000:4b:00.0 0000:57:00.0
 *   sudo ./e2e_smoke --cuda 0 --service 127.0.0.1:50051 --dev-id 0 --dev-id 1
 */

#include "../../coordinator/include/coordinator.h"
#include "../../coordinator/include/device.h"

#include "../../block_storage/include/block_storage.h"
#include "../../nvme_storage/include/host_fs_backed_nvme_storage.h"
#include "../../nvme_storage/include/nvme_storage.h"
#include "../../nvme_storage/include/nvme_file.h"
#include "../../memory/include/memory_subsystem.h"
#include "../../memory/include/memory_kind.h"
#include "../../memory/include/memory_region.h"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
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

constexpr uint32_t kNumShards     = 2;
constexpr uint32_t kLayers        = 4;
constexpr uint32_t kTensorSize    = 128u * 1024;
constexpr uint64_t kGpuFileSize   = (uint64_t)kNumShards * kLayers * kTensorSize;
constexpr std::size_t kRegionSize   = 1u << 20;
constexpr std::size_t kGranularity  = 128u * 1024;
constexpr std::size_t kNumSubSlices = kRegionSize / kGranularity;   // 8
constexpr const char* kGpuFileName  = "e2e_smoke_E2E";

uint64_t seed_of(uint32_t t, uint32_t round) {
    return (((uint64_t)round + 1) * 0x9E3779B97F4A7C15ULL) ^
           (((uint64_t)t + 1) * 0xC2B2AE3D27D4EB4FULL) ^ 0xC0FFEEULL;
}
void fill_pattern(uint8_t* b, uint32_t n, uint64_t s) {
    for (uint32_t i = 0; i < n; ++i) b[i] = (uint8_t)((s * 0xC2B2AE3D27D4EB4FULL + i) & 0xFF);
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

void usage(const char* p) {
    std::fprintf(stderr,
        "Usage:\n"
        "  IN_PROCESS (this process owns the NVMe):\n"
        "    %s [--cuda N] [--cap N] <PCI_BDF> <PCI_BDF>\n"
        "  SERVICE_CLIENT (attach via a running tutti_daemon):\n"
        "    %s [--cuda N] --service <endpoint> [--dev-id N --dev-id M]\n"
        "\n"
        "  --service <ep>   gRPC endpoint of tutti_daemon, e.g.\n"
        "                   127.0.0.1:50051 .  Switches to SERVICE_CLIENT.\n"
        "  --dev-id N       daemon-local device id to open (repeatable).\n"
        "                   Defaults to 0 1 when omitted in service mode.\n"
        "\n"
        "DESTRUCTIVE on unformatted disks; needs exactly 2 shards either\n"
        "way (2 BDFs in-process, or 2 daemon device ids in service mode).\n",
        p, p);
}

}  // namespace

int main(int argc, char** argv) {
    int cuda_dev = 0; uint32_t cap = 32;
    std::vector<std::string> bdfs;
    std::string service_endpoint;
    std::vector<int32_t> dev_ids;
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (!std::strcmp(a, "--cuda") && i + 1 < argc) cuda_dev = std::atoi(argv[++i]);
        else if (!std::strcmp(a, "--cap") && i + 1 < argc) cap = (uint32_t)std::atoi(argv[++i]);
        else if (!std::strcmp(a, "--service") && i + 1 < argc) service_endpoint = argv[++i];
        else if (!std::strcmp(a, "--dev-id") && i + 1 < argc) dev_ids.push_back((int32_t)std::atoi(argv[++i]));
        else if (!std::strcmp(a, "-h") || !std::strcmp(a, "--help")) { usage(argv[0]); return 0; }
        else if (a[0] != '-') bdfs.emplace_back(a);
        else { std::fprintf(stderr, "unknown arg: %s\n", a); usage(argv[0]); return 1; }
    }

    const bool service_mode = !service_endpoint.empty();
    if (service_mode) {
        if (dev_ids.empty()) dev_ids = {0, 1};       // default first two daemon devices
        if (dev_ids.size() != kNumShards) {
            std::fprintf(stderr, "service mode needs exactly %u --dev-id (got %zu)\n",
                         kNumShards, dev_ids.size());
            usage(argv[0]); return 1;
        }
    } else if (bdfs.size() != kNumShards) {
        std::fprintf(stderr, "in-process mode needs exactly %u BDFs (got %zu)\n",
                     kNumShards, bdfs.size());
        usage(argv[0]); return 1;
    }

    CUDA_OK(cudaSetDevice(cuda_dev));
    STEP_OK("cudaSetDevice(%d)", cuda_dev);

    // [2] one-call bootstrap of the whole stack via Coordinator.
    //     Same stack either way; cfg.mode picks the registry.
    tutti::CoordinatorConfig cfg;
    cfg.cuda_device                = cuda_dev;
    cfg.num_user_queues_per_device = 4;
    cfg.descriptor_format          = tutti::DescriptorFormat::PRP;
    cfg.max_entries_per_batch      = 64;
    if (service_mode) {
        cfg.mode              = tutti::CoordinatorMode::SERVICE_CLIENT;
        cfg.daemon_endpoint   = service_endpoint;
        cfg.daemon_device_ids = dev_ids;
    } else {
        cfg.mode           = tutti::CoordinatorMode::IN_PROCESS;
        cfg.pci_addrs      = bdfs;
        cfg.kernel_ioq_cap = cap;
    }

    tutti::Coordinator coord;
    if (!coord.bootstrap(cfg)) STEP_FAIL("Coordinator::bootstrap");
    STEP_OK("Coordinator::bootstrap [%s] (devices=%zu, one call brought up the stack)",
            service_mode ? "SERVICE_CLIENT" : "IN_PROCESS", coord.devices().size());

    // [3] pre-clean any straggler + open_gpu_file(CREATE).
    tutti::GpuFileSpec spec{};
    spec.name             = kGpuFileName;
    spec.total_size       = kGpuFileSize;
    spec.tensor_shape[0]  = kNumShards;
    spec.tensor_shape[1]  = kLayers;
    spec.tensor_shape[2]  = kTensorSize;
    spec.shard_placement  = { coord.devices()[0], coord.devices()[1] };

    if (tutti::GpuFile* old = coord.open_gpu_file(spec))
        (void)coord.delete_gpu_file(old, /*persist_now=*/true);

    tutti::GpuFile* gf = coord.open_gpu_file(
        spec, tutti::GPU_FILE_OPEN_CREATE | tutti::GPU_FILE_OPEN_EXCL);
    if (gf == nullptr) STEP_FAIL("open_gpu_file(CREATE)");
    STEP_OK("open_gpu_file(CREATE) '%s' (shards=%u, total=%llu)",
            kGpuFileName, kNumShards, (unsigned long long)kGpuFileSize);

    // [4] host-write seed-A patterns to shards.  O_DIRECT requires
    //     block-aligned buffers.
    std::vector<uint8_t*> patA(kNumSubSlices, nullptr);
    for (uint32_t t = 0; t < kNumSubSlices; ++t) {
        if (::posix_memalign(reinterpret_cast<void**>(&patA[t]), 4096, kTensorSize) != 0)
            STEP_FAIL("posix_memalign(patA[%u])", t);
        std::memset(patA[t], 0, kTensorSize);
        fill_pattern(patA[t], kTensorSize, seed_of(t, 0));
        const uint32_t sh = t % kNumShards;
        const uint64_t off = (uint64_t)(t / kNumShards) * kTensorSize;
        if (coord.storage()->write_blocking(gf->shards[sh], off,
                                            patA[t], kTensorSize)
            != (ssize_t)kTensorSize)
            STEP_FAIL("write_blocking seed-A t=%u", t);
    }
    for (uint32_t s = 0; s < kNumShards; ++s)
        if (!coord.storage()->sync(gf->shards[s])) STEP_FAIL("sync shard=%u", s);
    STEP_OK("host wrote %zu seed-A patterns + sync", kNumSubSlices);

    // [5] create the stream FIRST (R11: acquire_device_handle is now
    //     async/pooled and queues its H2D fill on a stream), then
    //     acquire_device_handle + allocate/register tensor.
    cudaStream_t stream = nullptr;
    CUDA_OK(cudaStreamCreate(&stream));

    tutti::GpuFileHandle* h = coord.acquire_device_handle(gf, stream);
    if (h == nullptr || h->d_shards_dev == nullptr)
        STEP_FAIL("acquire_device_handle / d_shards_dev null");

    tutti::MemoryRegion* tr =
        coord.allocate_device(kRegionSize, tutti::MemoryKind::DEVICE, cuda_dev);
    if (tr == nullptr) STEP_FAIL("allocate_device");
    {
        tutti::TensorRegistrationSpec ds{};
        ds.ptr = tr->device_ptr; ds.size = tr->size; ds.granularity = kGranularity;
        if (coord.register_tensor(ds) != tr) STEP_FAIL("register_tensor");
    }
    CUDA_OK(cudaMemset(tr->device_ptr, 0xAB, kRegionSize));
    STEP_OK("acquire_device_handle(stream) [async] + register_tensor(1 MiB / 128 KiB) "
            "+ memset(0xAB)");

    // [6] READ via Coordinator::submit_batch.  Same `stream` as the
    //     acquire above -- CUDA stream-ordering guarantees the H2D
    //     handle fill has completed before this kernel reads it, with
    //     no CPU-side wait in between.
    if (!coord.submit_batch(tr, h, /*file_byte_offset=*/0, /*is_read=*/true, stream))
        STEP_FAIL("submit_batch(read)");
    STEP_OK("coord.submit_batch(read) ok");

    // [7] verify device tensor == seed-A.
    {
        std::vector<uint8_t> got(kRegionSize, 0);
        CUDA_OK(cudaMemcpy(got.data(), tr->device_ptr, kRegionSize, cudaMemcpyDeviceToHost));
        for (uint32_t t = 0; t < kNumSubSlices; ++t)
            if (std::memcmp(got.data() + (uint64_t)t * kTensorSize,
                            patA[t], kTensorSize) != 0)
                STEP_FAIL("READ byte-compare mismatch at sub-slice %u", t);
    }
    // patA no longer needed -- free aligned buffers.
    for (auto p : patA) ::free(p);
    STEP_OK("READ verify: device tensor matches seed-A across %zu sub-slices",
            kNumSubSlices);

    // [8] WRITE: load seed-B into device tensor, submit write, verify shards.
    std::vector<std::vector<uint8_t>> patB(kNumSubSlices);
    for (uint32_t t = 0; t < kNumSubSlices; ++t) {
        patB[t].assign(kTensorSize, 0);
        fill_pattern(patB[t].data(), kTensorSize, seed_of(t, 1));
        CUDA_OK(cudaMemcpy(static_cast<uint8_t*>(tr->device_ptr) + (uint64_t)t * kTensorSize,
                           patB[t].data(), kTensorSize, cudaMemcpyHostToDevice));
    }
    if (!coord.submit_batch(tr, h, /*file_byte_offset=*/0, /*is_read=*/false, stream))
        STEP_FAIL("submit_batch(write)");
    STEP_OK("coord.submit_batch(write) ok");

    // read_blocking uses O_DIRECT (bypasses page cache), so no
    // posix_fadvise cache-drop is needed before host pread.
    {
        AlignedBuf got(kTensorSize);
        for (uint32_t t = 0; t < kNumSubSlices; ++t) {
            const uint32_t sh = t % kNumShards;
            const uint64_t off = (uint64_t)(t / kNumShards) * kTensorSize;
            if (coord.storage()->read_blocking(gf->shards[sh], off,
                                               got.data(), kTensorSize)
                != (ssize_t)kTensorSize)
                STEP_FAIL("read_blocking verify t=%u", t);
            if (std::memcmp(got.data(), patB[t].data(), kTensorSize) != 0)
                STEP_FAIL("WRITE byte-compare mismatch at sub-slice %u", t);
        }
    }
    STEP_OK("WRITE verify: shards match seed-B across %zu sub-slices", kNumSubSlices);

    // [9] teardown.  R11: release_device_handle is async -- queue it
    //     on `stream` BEFORE destroying that stream / freeing `tr` /
    //     deleting the GpuFile, then synchronize once so every
    //     deferred cudaFree the release triggers has actually run.
    coord.release_device_handle(h, stream);
    CUDA_OK(cudaStreamSynchronize(stream));
    cudaStreamDestroy(stream);
    coord.free(tr);
    if (!coord.delete_gpu_file(gf, /*persist_now=*/true)) STEP_FAIL("delete_gpu_file");
    coord.shutdown();
    STEP_OK("teardown: handle released (async, synced), tensor freed, gpu file "
            "deleted, coordinator shut down");

    std::fprintf(stderr,
        "\n=== e2e_smoke (devices=%u, sub_slices=%zu, total=%llu B): "
        "all %d steps passed ===\n",
        kNumShards, kNumSubSlices, (unsigned long long)kRegionSize, g_step);
    return 0;
}
