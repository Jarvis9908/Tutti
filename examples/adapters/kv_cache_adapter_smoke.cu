/**
 * kv_cache_adapter_smoke.cu -- end-to-end smoke for KvCacheIoAdapter.
 *
 * Exercises the generic io_engine's per-layer K/V handling through the
 * transparent runtime-owned handle model:
 *   - per-layer K/V byte-offset semantics live in adapters/kv_cache.
 *   - auto-chunking: max_entries_per_batch is set DELIBERATELY SMALL
 *       (2) while each per-layer batch flattens to 4 entries
 *       (kNumBlocks * {K,V}), so every layer submit is split into two
 *       engine batches by the adapter.
 *   - files are addressed by stable GpuFileId only; the Coordinator
 *       lazily acquires + caches the device handle (handle_for).  The
 *       test drops the cache mid-run and re-reads by id to prove the
 *       handle rebuilds from the persistent id alone.
 *   - coord.sync_file(id) is the durability point after writes.
 *
 * Scenario (KV-cache shaped):
 *   blocks   = 2  (each block == one GpuFile)
 *   layers   = 2
 *   kv_dim   = 2  (standard K/V split)
 *   shards   = 2
 *   tensor   = 64 KiB per (block, layer, K|V)
 *   GpuFile  : tensor_shape = [2, 2, 64 KiB], total = 256 KiB
 *
 * Striping (derived from the legacy formula, shards=2, kv_dim=2):
 *   K(layer l) -> shard0 @ l*64KiB ;  V(layer l) -> shard1 @ l*64KiB .
 *
 * Bring-up mode (same switch as e2e_smoke): defaults to IN_PROCESS;
 * pass --service <endpoint> to drive the SAME test through a running
 * tutti_daemon over SERVICE_CLIENT.
 *
 * Usage:
 *   IN_PROCESS:
 *     sudo ./kv_cache_adapter_smoke --cuda 0 --cap 32 0000:4b:00.0 0000:57:00.0
 *   SERVICE_CLIENT:
 *     sudo ./kv_cache_adapter_smoke --cuda 0 --service 127.0.0.1:50051 \
 *                                   --dev-id 0 --dev-id 1
 *
 * DESTRUCTIVE on unformatted disks; needs exactly 2 shards either way.
 */

#include "../../coordinator/include/coordinator.h"
#include "../../coordinator/include/device.h"

#include "../../adapters/kv_cache/include/kv_cache_io_adapter.h"

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

constexpr uint32_t kNumShards   = 2;
constexpr uint32_t kNumBlocks   = 2;
constexpr uint32_t kLayers      = 2;
constexpr uint32_t kKvDim       = 2;                       // standard K/V
constexpr uint32_t kTensorSize  = 64u * 1024;              // 64 KiB
constexpr uint64_t kGpuFileSize =
    (uint64_t)kLayers * kKvDim * kTensorSize;              // 256 KiB
// per-layer batch flattens to kNumBlocks * 2 (K+V) = 4 entries; cap at
// 2 to force the adapter to split each layer submit into 2 batches.
constexpr uint32_t kMaxEntries  = 2;

uint64_t seed_of(uint32_t b, uint32_t l, uint32_t kv, uint32_t round) {
    return (((uint64_t)round + 1) * 0x9E3779B97F4A7C15ULL) ^
           (((uint64_t)b + 1)     * 0xC2B2AE3D27D4EB4FULL) ^
           (((uint64_t)l + 1)     * 0xD6E8FEB86659FD93ULL) ^
           (((uint64_t)kv + 1)    * 0xA24BAED4963EE407ULL) ^ 0xC0FFEEULL;
}
void fill_pattern(uint8_t* p, uint32_t n, uint64_t s) {
    for (uint32_t i = 0; i < n; ++i)
        p[i] = (uint8_t)((s * 0xC2B2AE3D27D4EB4FULL + i) & 0xFF);
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
};

// shard / shard-offset for (layer, kv) under the legacy formula with
// shards=2, kv_dim=2:  K -> shard0, V -> shard1, both @ layer*tensor.
uint32_t shard_of(uint32_t kv)    { return kv % kNumShards; }
uint64_t shard_off_of(uint32_t l) { return (uint64_t)l * kTensorSize; }

void usage(const char* p) {
    std::fprintf(stderr,
        "Usage:\n"
        "  IN_PROCESS (this process owns the NVMe):\n"
        "    %s [--cuda N] [--cap N] <PCI_BDF> <PCI_BDF>\n"
        "  SERVICE_CLIENT (attach via a running tutti_daemon):\n"
        "    %s [--cuda N] --service <endpoint> [--dev-id N --dev-id M]\n"
        "\n"
        "DESTRUCTIVE on unformatted disks; needs exactly 2 shards.\n",
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
        if (dev_ids.empty()) dev_ids = {0, 1};
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

    // [2] one-call bootstrap; cfg.mode picks the registry (same stack).
    tutti::CoordinatorConfig cfg;
    cfg.cuda_device                = cuda_dev;
    cfg.num_user_queues_per_device = 4;
    cfg.descriptor_format          = tutti::DescriptorFormat::PRP;
    cfg.max_entries_per_batch      = kMaxEntries;        // force adapter chunking
    cfg.handle_l1_capacity          = kNumBlocks;         // exactly the working set
    cfg.handle_l2_capacity          = kNumBlocks;         // same -- also exercise L2 eviction
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
    STEP_OK("Coordinator::bootstrap [%s] (devices=%zu)",
            service_mode ? "SERVICE_CLIENT" : "IN_PROCESS", coord.devices().size());

    // [3] create kNumBlocks GpuFiles; keep only their stable ids.
    std::vector<tutti::GpuFileId> file_ids(kNumBlocks, 0);
    std::vector<tutti::GpuFile*>  gf(kNumBlocks, nullptr);   // for host-side seed/verify
    for (uint32_t b = 0; b < kNumBlocks; ++b) {
        std::string name = "kv_adapter_smoke_b" + std::to_string(b);
        tutti::GpuFileSpec spec{};
        spec.name            = name;
        spec.total_size      = kGpuFileSize;
        spec.tensor_shape[0] = kNumShards;
        spec.tensor_shape[1] = kLayers * kKvDim / kNumShards;   // = 2
        spec.tensor_shape[2] = kTensorSize;
        spec.shard_placement = { coord.devices()[0], coord.devices()[1] };

        if (tutti::GpuFile* old = coord.open_gpu_file(spec))
            (void)coord.delete_gpu_file(old, /*persist_now=*/true);
        gf[b] = coord.open_gpu_file(
            spec, tutti::GPU_FILE_OPEN_CREATE | tutti::GPU_FILE_OPEN_EXCL);
        if (gf[b] == nullptr) STEP_FAIL("open_gpu_file(CREATE) b=%u", b);
        file_ids[b] = gf[b]->id;
    }
    STEP_OK("created %u GpuFiles (ids only to caller; handles stay runtime-owned)",
            kNumBlocks);

    // [4] host-write seed-A to every (block, layer, K|V) location.
    // O_DIRECT requires block-aligned buffers (posix_memalign per leaf).
    auto seed_shards = [&](uint32_t round,
                           std::vector<std::vector<std::vector<uint8_t*>>>& pat) {
        pat.assign(kNumBlocks,
            std::vector<std::vector<uint8_t*>>(kLayers,
                std::vector<uint8_t*>(kKvDim, nullptr)));
        for (uint32_t b = 0; b < kNumBlocks; ++b)
            for (uint32_t l = 0; l < kLayers; ++l)
                for (uint32_t kv = 0; kv < kKvDim; ++kv) {
                    if (::posix_memalign(reinterpret_cast<void**>(&pat[b][l][kv]),
                                         4096, kTensorSize) != 0)
                        STEP_FAIL("posix_memalign(pat[%u][%u][%u])", b, l, kv);
                    std::memset(pat[b][l][kv], 0, kTensorSize);
                    fill_pattern(pat[b][l][kv], kTensorSize, seed_of(b, l, kv, round));
                    const uint32_t sh  = shard_of(kv);
                    const uint64_t off = shard_off_of(l);
                    if (coord.storage()->write_blocking(gf[b]->shards[sh], off,
                                                        pat[b][l][kv], kTensorSize)
                        != (ssize_t)kTensorSize)
                        STEP_FAIL("write_blocking seed b=%u l=%u kv=%u", b, l, kv);
                }
        for (uint32_t b = 0; b < kNumBlocks; ++b)
            for (uint32_t s = 0; s < kNumShards; ++s)
                if (!coord.storage()->sync(gf[b]->shards[s]))
                    STEP_FAIL("sync b=%u shard=%u", b, s);
    };
    std::vector<std::vector<std::vector<uint8_t*>>> patA;
    seed_shards(/*round=*/0, patA);
    STEP_OK("host wrote seed-A across %u blocks x %u layers x %u kv + sync",
            kNumBlocks, kLayers, kKvDim);

    // [5] allocate/register one device tensor per (block, layer, K|V).
    //     No explicit acquire_device_handle -- the adapter/coordinator
    //     resolve + cache handles from file_ids transparently.
    std::vector<std::vector<std::vector<tutti::MemoryRegion*>>> reg(
        kNumBlocks,
        std::vector<std::vector<tutti::MemoryRegion*>>(kLayers,
            std::vector<tutti::MemoryRegion*>(kKvDim, nullptr)));
    for (uint32_t b = 0; b < kNumBlocks; ++b)
        for (uint32_t l = 0; l < kLayers; ++l)
            for (uint32_t kv = 0; kv < kKvDim; ++kv) {
                tutti::MemoryRegion* tr =
                    coord.allocate_device(kTensorSize, tutti::MemoryKind::DEVICE, cuda_dev);
                if (tr == nullptr) STEP_FAIL("allocate_device b=%u l=%u kv=%u", b, l, kv);
                tutti::TensorRegistrationSpec ds{};
                ds.ptr = tr->device_ptr; ds.size = tr->size; ds.granularity = kTensorSize;
                if (coord.register_tensor(ds) != tr)
                    STEP_FAIL("register_tensor b=%u l=%u kv=%u", b, l, kv);
                CUDA_OK(cudaMemset(tr->device_ptr, 0xAB, kTensorSize));
                reg[b][l][kv] = tr;
            }
    STEP_OK("registered %u device tensors (gran=%u, memset 0xAB)",
            kNumBlocks * kLayers * kKvDim, kTensorSize);

    // [6] build the KV adapter (tensor_size == GpuFile tensor_shape[2]).
    tutti::KvCacheIoAdapter kv(coord, kTensorSize, /*use_mla=*/false);

    cudaStream_t stream = nullptr;
    CUDA_OK(cudaStreamCreate(&stream));

    auto k_list = [&](uint32_t l) {
        std::vector<tutti::MemoryRegion*> v;
        for (uint32_t b = 0; b < kNumBlocks; ++b) v.push_back(reg[b][l][0]);
        return v;
    };
    auto v_list = [&](uint32_t l) {
        std::vector<tutti::MemoryRegion*> v;
        for (uint32_t b = 0; b < kNumBlocks; ++b) v.push_back(reg[b][l][1]);
        return v;
    };

    auto verify_device_matches =
        [&](const std::vector<std::vector<std::vector<uint8_t*>>>& pat,
            const char* tag) {
            std::vector<uint8_t> got(kTensorSize, 0);
            for (uint32_t b = 0; b < kNumBlocks; ++b)
                for (uint32_t l = 0; l < kLayers; ++l)
                    for (uint32_t kvi = 0; kvi < kKvDim; ++kvi) {
                        CUDA_OK(cudaMemcpy(got.data(), reg[b][l][kvi]->device_ptr,
                                           kTensorSize, cudaMemcpyDeviceToHost));
                        if (std::memcmp(got.data(), pat[b][l][kvi], kTensorSize) != 0)
                            STEP_FAIL("%s mismatch b=%u l=%u kv=%u", tag, b, l, kvi);
                    }
        };

    // [7] READ each layer BY ID (handles resolved + cached transparently;
    //     auto-chunked: 4 entries / cap 2).
    for (uint32_t l = 0; l < kLayers; ++l)
        if (!kv.batched_read((int)l, k_list(l), v_list(l), file_ids, stream))
            STEP_FAIL("adapter.batched_read(by id) layer=%u", l);
    verify_device_matches(patA, "READ");
    STEP_OK("READ by id verify: %u device tensors match seed-A (handles lazy-acquired)",
            kNumBlocks * kLayers * kKvDim);
    // patA no longer needed -- free aligned buffers.
    for (auto& bb : patA) for (auto& ll : bb) for (auto p : ll) ::free(p);

    // [8] WRITE seed-B into device tensors, adapter write BY ID, then
    //     sync_file(id) as the durability point.
    std::vector<std::vector<std::vector<uint8_t*>>> patB(
        kNumBlocks,
        std::vector<std::vector<uint8_t*>>(kLayers,
            std::vector<uint8_t*>(kKvDim, nullptr)));
    for (uint32_t b = 0; b < kNumBlocks; ++b)
        for (uint32_t l = 0; l < kLayers; ++l)
            for (uint32_t kvi = 0; kvi < kKvDim; ++kvi) {
                if (::posix_memalign(reinterpret_cast<void**>(&patB[b][l][kvi]),
                                     4096, kTensorSize) != 0)
                    STEP_FAIL("posix_memalign(patB[%u][%u][%u])", b, l, kvi);
                std::memset(patB[b][l][kvi], 0, kTensorSize);
                fill_pattern(patB[b][l][kvi], kTensorSize, seed_of(b, l, kvi, /*round=*/1));
                CUDA_OK(cudaMemcpy(reg[b][l][kvi]->device_ptr, patB[b][l][kvi],
                                   kTensorSize, cudaMemcpyHostToDevice));
            }
    for (uint32_t l = 0; l < kLayers; ++l)
        if (!kv.batched_write((int)l, k_list(l), v_list(l), file_ids, stream))
            STEP_FAIL("adapter.batched_write(by id) layer=%u", l);
    for (uint32_t b = 0; b < kNumBlocks; ++b)
        if (!coord.sync_file(file_ids[b], stream)) STEP_FAIL("sync_file id=%u", file_ids[b]);
    STEP_OK("WRITE by id over %u layers + sync_file per block (durability point)", kLayers);

    // verify shards via host pread.  read_blocking uses O_DIRECT
    // (bypasses page cache), so no posix_fadvise cache-drop is needed.
    {
        AlignedBuf got(kTensorSize);
        for (uint32_t b = 0; b < kNumBlocks; ++b)
            for (uint32_t l = 0; l < kLayers; ++l)
                for (uint32_t kvi = 0; kvi < kKvDim; ++kvi) {
                    const uint32_t sh  = shard_of(kvi);
                    const uint64_t off = shard_off_of(l);
                    if (coord.storage()->read_blocking(gf[b]->shards[sh], off,
                                                       got.data(), kTensorSize)
                        != (ssize_t)kTensorSize)
                        STEP_FAIL("read_blocking verify b=%u l=%u kv=%u", b, l, kvi);
                    if (std::memcmp(got.data(), patB[b][l][kvi], kTensorSize) != 0)
                        STEP_FAIL("WRITE mismatch b=%u l=%u kv=%u", b, l, kvi);
                }
    }
    STEP_OK("WRITE verify: all shards match seed-B");

    // [9] TRANSPARENT REBUILD: drop every cached handle, scribble the
    //     device tensors, then read BY ID again.  The Coordinator must
    //     re-acquire each handle from the stable id alone.
    //     R11: drop is async on `stream`; the sync cudaMemsets right
    //     below implicitly wait for it (plain cudaMemset/cudaMemcpy
    //     always sync against every stream's outstanding work).
    coord.drop_cached_handles(stream);
    for (uint32_t b = 0; b < kNumBlocks; ++b)
        for (uint32_t l = 0; l < kLayers; ++l)
            for (uint32_t kvi = 0; kvi < kKvDim; ++kvi)
                CUDA_OK(cudaMemset(reg[b][l][kvi]->device_ptr, 0xCD, kTensorSize));
    for (uint32_t l = 0; l < kLayers; ++l)
        if (!kv.batched_read((int)l, k_list(l), v_list(l), file_ids, stream))
            STEP_FAIL("adapter.batched_read(after drop) layer=%u", l);
    verify_device_matches(patB, "REBUILD-READ");
    STEP_OK("handle rebuild verify: dropped cache, re-read by id, tensors match seed-B");
    // patB no longer needed -- free aligned buffers.
    for (auto& bb : patB) for (auto& ll : bb) for (auto p : ll) ::free(p);

    // [10] teardown.  No explicit handle release: delete_gpu_file evicts
    //      the cached handle, shutdown() releases any remainder.
    cudaStreamDestroy(stream);
    for (uint32_t b = 0; b < kNumBlocks; ++b)
        for (uint32_t l = 0; l < kLayers; ++l)
            for (uint32_t kvi = 0; kvi < kKvDim; ++kvi)
                coord.free(reg[b][l][kvi]);
    for (uint32_t b = 0; b < kNumBlocks; ++b)
        if (!coord.delete_gpu_file(gf[b], /*persist_now=*/true))
            STEP_FAIL("delete_gpu_file b=%u", b);
    coord.shutdown();
    STEP_OK("teardown complete (handles released transparently)");

    std::fprintf(stderr,
        "\n=== kv_cache_adapter_smoke [%s] (blocks=%u layers=%u kv=%u, "
        "cap=%u): all %d steps passed ===\n",
        service_mode ? "SERVICE_CLIENT" : "IN_PROCESS",
        kNumBlocks, kLayers, kKvDim, kMaxEntries, g_step);
    return 0;
}
