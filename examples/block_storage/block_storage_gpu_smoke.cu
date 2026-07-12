/**
 * block_storage_gpu_smoke.cu -- R6.3 end-to-end GPU device-side
 * submit through HostFsBackedBlockStorage.
 *
 * Layer: block_storage tests.
 *
 * What this verifies (all in one run):
 *   1. block_storage::acquire_device_handle(GpuFile*) really
 *      cudaMallocs one NvmeFileDeviceHandle per shard on the right
 *      GPU, in the right order, and returns a host-resident
 *      GpuFileHandle that owns them.
 *   2. The "host stages per-IO context, kernel reads NvmeFileDeviceHandle*
 *      directly" submission model that R8 io_engine will use --
 *      no GPU-resident shard-table indirection.
 *   3. Round-robin shard placement via gpu_file_resolve produces
 *      consistent results host->GPU (host write_blocking byte-for-byte
 *      matches the GPU read into device buffer).
 *
 * Layout knobs (chosen to stay within R5b's single-page PRP1
 * envelope -- multi-page PRP-list builds belong to R7 memory/):
 *
 *     num_shards         = 2          (first 2 PCI BDFs from CLI)
 *     layers_per_shard   = 128        (tensor_shape[1])
 *     tensor_size        = 4 KiB      (= 1 NVMe block, prp2 = 0)
 *     total_size         = 1 MiB
 *     io_count_per_run   = 256        (round-robin, kNumShards * kLayers)
 *
 * Per-shard GPU buffers:
 *
 *     shard buffer bytes = layers_per_shard * tensor_size = 512 KiB
 *     cudaMalloc + nvm_dma_map_data_device on THAT shard's controller.
 *     Each shard's NVMe controller has its own IOVA space, so the
 *     two buffers are mapped independently; their ioaddrs[] arrays
 *     are NOT comparable across shards.
 *
 * Step list:
 *
 *     [ 1] cuda prime + cudaSetDevice
 *     [ 2] LocalNvmeDirectRegistry::Open() with build_queue_group=true,
 *          num_user_queues=4, so each device's libnvm Controller has
 *          a GPU-resident d_qps[] -- required for R5b submit_read_one.
 *     [ 3] HostFsBackedNvmeStorage::bootstrap()
 *     [ 4] HostFsBackedBlockStorage::bootstrap()
 *     [ 5] pre-cleanup: drop "gpu_smoke_blk_*" stragglers
 *     [ 6] open_gpu_file(CREATE) with the layout above (one GpuFile, both
 *          shards across the two devices in CLI order)
 *     [ 7] host write_blocking a per-tensor pattern via
 *          gpu_file_resolve + INvmeStorage::write_blocking on the
 *          resolved shard NvmeFile.  Sync each shard.
 *     [ 8] per-shard cudaMalloc(layers*tensor_size) +
 *          cudaMemset(0) + nvm_dma_map_data_device on shard's ctrl.
 *     [ 9] block_storage->acquire_device_handle(file)
 *          -> GpuFileHandle with d_shards_host[shard_idx]
 *             pointing at GPU-resident NvmeFileDeviceHandle*s
 *     [10] host-side stage GPUIoContext array on the host, with
 *          per-IO (NvmeFileDeviceHandle*, prp1, shard_byte_off, len)
 *          resolved via gpu_file_resolve.  cudaMemcpy to device.
 *     [11] launch read kernel <<<ceil(N/32), 32>>>; each thread
 *          calls submit_read_one with the ctx it owns.  No shard
 *          indirection -- ctx already carries the resolved handle.
 *     [12] cudaDeviceSynchronize + check no late launch errors.
 *     [13] cudaMemcpy each per-shard buffer back to host;
 *          byte-compare against the patterns from [7].
 *     [14] release_device_handle(handle); per-shard nvm_dma_unmap +
 *          cudaFree.
 *     [15] delete_gpu_file
 *     [16] block_storage shutdown + nvme_storage shutdown
 *     [17] registry close
 *
 * Multi-NVMe invocation (need at least 2 BDFs because num_shards=2):
 *   sudo ./block_storage_gpu_smoke --gpu 0 --cuda 0 --cap 32 \
 *        0000:4b:00.0 0000:57:00.0
 *
 * Extra BDFs beyond the first 2 are accepted; they bootstrap normally
 * (so the gpu_file_log mirroring is exercised at >2 mirrors) but do
 * not host shards for this smoke.
 */

#include "host_fs_backed_block_storage.h"
#include "block_storage.h"
#include "gpu_file_resolve.h"

#include "host_fs_backed_nvme_storage.h"
#include "nvme_storage.h"
#include "nvme_file.h"
#include "nvme_file_device_handle.h"
#include "nvme_storage_device.cuh"

#include "../common/registry_cli.h"
#include "../../device_manager/include/local_nvme_device.h"
#include "../../coordinator/include/device.h"

#include <nvm_ctrl.h>
#include <nvm_dma.h>

#include <cuda_runtime.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Step harness
// ---------------------------------------------------------------------------
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

void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage:\n"
        "  IN_PROCESS: %s [--gpu N] [--cuda N] [--cap N] <PCI_BDF> <PCI_BDF> [...]\n"
        "  SERVICE_CLIENT: %s [--cuda N] --service <endpoint> --dev-id N --dev-id M\n"
        "  e.g.: %s --gpu 0 --cuda 0 --cap 32 0000:4b:00.0 0000:57:00.0\n"
        "        %s --cuda 0 --service 127.0.0.1:50051 --dev-id 0 --dev-id 1\n"
        "DESTRUCTIVE on unformatted disks (mkfs.ext4 -F via nvme_storage).\n"
        "Needs at least 2 devices (num_shards = 2).\n",
        prog, prog, prog, prog);
}

void prime_cuda(int cuda_dev) {
    cudaError_t cerr = cudaFree(0);
    if (cerr != cudaSuccess && cerr != cudaErrorInvalidValue) {
        STEP_FAIL("cuda driver prime failed: %s", cudaGetErrorString(cerr));
    }
    (void)cudaGetLastError();
    cerr = cudaSetDevice(cuda_dev);
    if (cerr != cudaSuccess) STEP_FAIL("cudaSetDevice(%d): %s",
                                       cuda_dev, cudaGetErrorString(cerr));
    STEP_OK("cudaSetDevice(%d)", cuda_dev);
}

// ---------------------------------------------------------------------------
// Layout knobs
// ---------------------------------------------------------------------------
constexpr uint32_t kNumShards    = 2;
constexpr uint32_t kLayers       = 128;
constexpr uint32_t kTensorSize   = 4096;                 // 4 KiB, 1 PRP page
constexpr uint64_t kGpuFileSize  =
    (uint64_t)kNumShards * kLayers * kTensorSize;        // 1 MiB
constexpr uint64_t kPerShardBuf  = (uint64_t)kLayers * kTensorSize;  // 512 KiB
constexpr uint32_t kNumIos       = kNumShards * kLayers;  // 256
constexpr const char* kGpuFileName = "gpu_smoke_blk_0";
constexpr const char* kPrefix      = "gpu_smoke_blk_";

uint64_t pattern_seed(uint32_t tensor_idx) {
    return ((uint64_t)tensor_idx + 1u) * 0x9E3779B97F4A7C15ULL ^ 0xDEADBEEFCAFEULL;
}

void fill_pattern(uint8_t* buf, uint64_t seed) {
    for (uint32_t i = 0; i < kTensorSize; ++i) {
        buf[i] = (uint8_t)((seed * 0xC2B2AE3D27D4EB4FULL + i) & 0xFFu);
    }
}

// ---------------------------------------------------------------------------
// Per-IO context + kernel.  Mirrors the shape R8 io_engine will
// adopt: ctx carries an already-resolved NvmeFileDeviceHandle*
// (via host stage), so the kernel performs no shard-table lookup.
// ---------------------------------------------------------------------------
struct R6GpuIoCtx {
    const tutti::NvmeFileDeviceHandle* dh;          // GPU-resident
    uint64_t                           prp1;        // shard ctrl IOVA
    uint64_t                           prp2;        // 0 for single-page IO
    uint64_t                           shard_off;   // bytes into shard
    uint64_t                           nbytes;      // 4 KiB here
};

__global__ void block_storage_read_kernel(const R6GpuIoCtx* ctx, uint32_t n) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    const auto& c = ctx[tid];
    tutti::submit_read_one(c.dh, c.prp1, c.prp2, c.shard_off, c.nbytes);
}

// ---------------------------------------------------------------------------
// Idempotency helper (R6.2-style)
// ---------------------------------------------------------------------------
void wipe_stragglers(tutti::IBlockStorage& bs) {
    std::size_t n = 0;
    for (const auto& nm : bs.list_gpu_file_names()) {
        if (nm.rfind(kPrefix, 0) != 0) continue;
        tutti::GpuFileSpec spec{};
        spec.name = nm;
        tutti::GpuFile* gf = bs.open_gpu_file(spec);   // EXISTING: only spec.name matters
        if (gf == nullptr) continue;
        if (bs.delete_gpu_file(gf, /*persist_now=*/false)) ++n;
    }
    if (n > 0) (void)bs.flush_metadata();
    if (n > 0) {
        std::fprintf(stderr,
            "[block_storage] pre-cleanup: removed %zu '%s*' "
            "GpuFile straggler(s)\n", n, kPrefix);
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    tutti_examples::RegistryCliOptions ropt;
    ropt.num_user_queues = 4;

    for (int i = 1; i < argc; ) {
        if (tutti_examples::parse_registry_cli_arg(ropt, argc, argv, i)) { ++i; continue; }
        const char* a = argv[i];
        if (std::strcmp(a, "--gpu") == 0 && i + 1 < argc) {
            i += 2;   // unused; kept for CLI compat with sibling smokes
        } else if (std::strcmp(a, "-h") == 0 || std::strcmp(a, "--help") == 0) {
            usage(argv[0]); return 0;
        } else if (a[0] != '-') {
            ropt.pci_addrs.emplace_back(a); ++i;
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", a);
            usage(argv[0]); return 1;
        }
    }
    if (!tutti_examples::validate_registry_cli(ropt, argv[0])) { usage(argv[0]); return 1; }
    const std::size_t n_dev_req = ropt.service_mode ? ropt.dev_ids.size() : ropt.pci_addrs.size();
    if (n_dev_req < kNumShards) {
        std::fprintf(stderr,
            "need at least %u devices (got %zu) -- num_shards = %u\n",
            kNumShards, n_dev_req, kNumShards);
        usage(argv[0]); return 1;
    }

    // [1]
    prime_cuda(ropt.cuda_dev);

    // [2] registry up -- build_queue_group = true (need d_qps for submit).
    //     IN_PROCESS or SERVICE_CLIENT, mode-agnostic past this point.
    auto reg = tutti_examples::open_registry(ropt, /*build_queue_group=*/true);
    if (reg.ptr == nullptr) STEP_FAIL("registry Open()");
    STEP_OK("registry up: mode=%s n=%zu (build_queue_group=true, q/dev=%u)",
            ropt.service_mode ? "service" : "direct", n_dev_req, ropt.num_user_queues);

    const std::size_t N_dev = reg.ptr->device_count();
    std::vector<const tutti::Device*> devices;
    devices.reserve(N_dev);
    for (std::size_t i = 0; i < N_dev; ++i) {
        const tutti::Device* d = reg.ptr->device_at(i);
        if (d == nullptr) STEP_FAIL("device_at(%zu) is null", i);
        devices.push_back(d);
    }

    // [3] nvme_storage bootstrap
    auto storage = std::make_unique<tutti::HostFsBackedNvmeStorage>();
    if (!storage->bootstrap(devices)) STEP_FAIL("storage.bootstrap");
    STEP_OK("nvme_storage bootstrap (devices=%zu)", N_dev);

    // [4] block_storage bootstrap
    auto bs = std::make_unique<tutti::HostFsBackedBlockStorage>();
    if (!bs->bootstrap(storage.get(), devices)) STEP_FAIL("block_storage.bootstrap");
    STEP_OK("block_storage bootstrap (entries=%zu)",
            bs->list_gpu_file_names().size());

    // [5] pre-cleanup
    wipe_stragglers(*bs);
    STEP_OK("pre-cleanup");

    // [6] open_gpu_file(CREATE) -- shards on devices[0] and devices[1]
    tutti::GpuFileSpec spec{};
    spec.name              = kGpuFileName;
    spec.total_size        = kGpuFileSize;
    spec.tensor_shape[0]   = kNumShards;
    spec.tensor_shape[1]   = kLayers;
    spec.tensor_shape[2]   = kTensorSize;
    spec.shard_placement   = { devices[0], devices[1] };

    tutti::GpuFile* gf = bs->open_gpu_file(
        spec, tutti::GPU_FILE_OPEN_CREATE | tutti::GPU_FILE_OPEN_EXCL);
    if (gf == nullptr) STEP_FAIL("open_gpu_file(CREATE) '%s'", kGpuFileName);
    if (gf->shards.size() != kNumShards)
        STEP_FAIL("created GpuFile has %zu shards (want %u)",
                  gf->shards.size(), kNumShards);
    STEP_OK("open_gpu_file(CREATE) '%s' total=%llu shards=%u tensor_size=%u",
            kGpuFileName,
            (unsigned long long)kGpuFileSize, kNumShards, kTensorSize);

    // [7] host write per-tensor patterns via gpu_file_resolve.
    // O_DIRECT requires block-aligned buffers.
    std::vector<uint8_t*> patterns(kNumIos, nullptr);
    for (uint32_t t = 0; t < kNumIos; ++t) {
        if (::posix_memalign(reinterpret_cast<void**>(&patterns[t]),
                             4096, kTensorSize) != 0)
            STEP_FAIL("posix_memalign(patterns[%u])", t);
        fill_pattern(patterns[t], pattern_seed(t));
        const uint64_t global_off = (uint64_t)t * kTensorSize;
        uint32_t shard_idx     = 0;
        uint64_t shard_byte_off = 0;
        tutti::gpu_file_resolve(kTensorSize, kNumShards, global_off,
                                &shard_idx, &shard_byte_off);
        if (shard_idx >= kNumShards)
            STEP_FAIL("resolve produced shard_idx=%u", shard_idx);
        ssize_t wr = storage->write_blocking(gf->shards[shard_idx],
                                             shard_byte_off,
                                             patterns[t],
                                             kTensorSize);
        if (wr != (ssize_t)kTensorSize)
            STEP_FAIL("write_blocking(shard=%u off=%llu) wr=%zd",
                      shard_idx, (unsigned long long)shard_byte_off, wr);
    }
    for (uint32_t s = 0; s < kNumShards; ++s) {
        if (!storage->sync(gf->shards[s])) STEP_FAIL("sync(shard=%u)", s);
    }
    STEP_OK("host wrote %u tensor pattern(s) (%llu bytes total) + sync",
            kNumIos, (unsigned long long)kGpuFileSize);

    // [8] per-shard cudaMalloc + nvm_dma_map_data_device on shard's
    //     own controller.  Each map is independent; ioaddrs[] from
    //     shard A is NOT valid in shard B's controller IOVA space.
    struct ShardBuf {
        void*       devptr  = nullptr;
        nvm_dma_t*  dma     = nullptr;          // owned, must nvm_dma_unmap
    };
    std::vector<ShardBuf> bufs(kNumShards);
    for (uint32_t s = 0; s < kNumShards; ++s) {
        const tutti::Device* dev = devices[s];
        auto* lnd = static_cast<tutti::LocalNvmeDevice*>(dev->backend_private);
        if (lnd == nullptr || lnd->ctrl == nullptr)
            STEP_FAIL("shard %u: no LocalNvmeDevice/ctrl", s);

        CUDA_OK(cudaMalloc(&bufs[s].devptr, kPerShardBuf));
        CUDA_OK(cudaMemset(bufs[s].devptr, 0, kPerShardBuf));

        int rc = nvm_dma_map_data_device(&bufs[s].dma,
                                         lnd->ctrl,
                                         bufs[s].devptr,
                                         kPerShardBuf);
        if (rc != 0 || bufs[s].dma == nullptr)
            STEP_FAIL("nvm_dma_map_data_device(shard=%u) rc=%d", s, rc);
        if (bufs[s].dma->n_ioaddrs == 0)
            STEP_FAIL("shard %u DMA mapping has 0 ioaddrs", s);
        if (bufs[s].dma->page_size != kTensorSize)
            STEP_FAIL("shard %u DMA page_size=%zu != tensor_size=%u",
                      s, (size_t)bufs[s].dma->page_size, kTensorSize);
        if (bufs[s].dma->n_ioaddrs * (size_t)bufs[s].dma->page_size <
            (size_t)kPerShardBuf)
            STEP_FAIL("shard %u DMA mapping covers only %zu of %llu bytes",
                      s,
                      bufs[s].dma->n_ioaddrs * (size_t)bufs[s].dma->page_size,
                      (unsigned long long)kPerShardBuf);
    }
    STEP_OK("cudaMalloc(%llu) + nvm_dma_map_data_device per shard "
            "(ioaddrs[shard0][0]=0x%llx, ioaddrs[shard1][0]=0x%llx)",
            (unsigned long long)kPerShardBuf,
            (unsigned long long)bufs[0].dma->ioaddrs[0],
            (unsigned long long)bufs[1].dma->ioaddrs[0]);

    // [9] configure the async handle pool (R11), then
    //     block_storage->acquire_device_handle.  Stream = default
    //     (nullptr/0) -- this smoke has no dedicated stream.
    if (!bs->configure_handle_pool(1, 1)) STEP_FAIL("configure_handle_pool");
    tutti::GpuFileHandle* h = bs->acquire_device_handle(gf, /*stream=*/nullptr);
    if (h == nullptr) STEP_FAIL("acquire_device_handle");
    if (h->num_shards != kNumShards)
        STEP_FAIL("handle->num_shards=%u (want %u)",
                  h->num_shards, kNumShards);
    if (h->d_shards_host.size() != kNumShards)
        STEP_FAIL("handle->d_shards_host.size()=%zu (want %u)",
                  h->d_shards_host.size(), kNumShards);
    for (uint32_t s = 0; s < kNumShards; ++s) {
        if (h->d_shards_host[s] == nullptr)
            STEP_FAIL("handle->d_shards_host[%u] == nullptr", s);
        // Verify the pointer is in GPU memory (not host).
        cudaPointerAttributes attr{};
        cudaError_t cerr = cudaPointerGetAttributes(&attr, h->d_shards_host[s]);
        if (cerr != cudaSuccess)
            STEP_FAIL("cudaPointerGetAttributes(d_shards_host[%u]) failed: %s",
                      s, cudaGetErrorString(cerr));
        if (attr.type != cudaMemoryTypeDevice)
            STEP_FAIL("d_shards_host[%u] is not GPU memory (type=%d)",
                      s, (int)attr.type);
    }
    STEP_OK("acquire_device_handle: %u shard handle(s), all GPU-resident",
            h->num_shards);

    // [10] host-stage GPUIoContext array.  For each tensor in the
    //      GpuFile we build one IO targeting the resolved shard's
    //      already-acquired NvmeFileDeviceHandle*.
    std::vector<R6GpuIoCtx> h_ctx(kNumIos);
    for (uint32_t t = 0; t < kNumIos; ++t) {
        const uint64_t global_off = (uint64_t)t * kTensorSize;
        uint32_t shard_idx     = 0;
        uint64_t shard_byte_off = 0;
        tutti::gpu_file_resolve(kTensorSize, kNumShards, global_off,
                                &shard_idx, &shard_byte_off);
        // PRP1 is the IOVA of the page that backs offset
        // (shard_byte_off, shard_byte_off + kTensorSize) in the
        // shard's own buffer.  Page size == tensor size here,
        // so each tensor maps to exactly one ioaddr.
        const uint32_t page_idx_in_shard =
            (uint32_t)(shard_byte_off / kTensorSize);
        h_ctx[t].dh         = h->d_shards_host[shard_idx];
        h_ctx[t].prp1       = bufs[shard_idx].dma->ioaddrs[page_idx_in_shard];
        h_ctx[t].prp2       = 0;     // single-page IO
        h_ctx[t].shard_off  = shard_byte_off;
        h_ctx[t].nbytes     = kTensorSize;
    }
    R6GpuIoCtx* d_ctx = nullptr;
    CUDA_OK(cudaMalloc(&d_ctx, sizeof(R6GpuIoCtx) * kNumIos));
    CUDA_OK(cudaMemcpy(d_ctx, h_ctx.data(),
                       sizeof(R6GpuIoCtx) * kNumIos, cudaMemcpyHostToDevice));
    STEP_OK("host-staged %u GPUIoContext entries (cross-shard)", kNumIos);

    // [11] launch read kernel
    constexpr uint32_t kBlockDim = 32;
    const     uint32_t kGridDim  = (kNumIos + kBlockDim - 1) / kBlockDim;
    block_storage_read_kernel<<<kGridDim, kBlockDim>>>(d_ctx, kNumIos);
    {
        cudaError_t cerr = cudaGetLastError();
        if (cerr != cudaSuccess)
            STEP_FAIL("kernel launch failed: %s", cudaGetErrorString(cerr));
    }
    STEP_OK("launched submit_read_one kernel grid=%u block=%u (n=%u)",
            kGridDim, kBlockDim, kNumIos);

    // [12] sync
    CUDA_OK(cudaDeviceSynchronize());
    {
        cudaError_t cerr = cudaGetLastError();
        if (cerr != cudaSuccess)
            STEP_FAIL("kernel reported error: %s", cudaGetErrorString(cerr));
    }
    STEP_OK("cudaDeviceSynchronize after kernel");

    // [13] cudaMemcpy each shard buffer back, byte-compare against patterns
    int total_mismatch = 0;
    for (uint32_t t = 0; t < kNumIos; ++t) {
        const uint64_t global_off = (uint64_t)t * kTensorSize;
        uint32_t shard_idx     = 0;
        uint64_t shard_byte_off = 0;
        tutti::gpu_file_resolve(kTensorSize, kNumShards, global_off,
                                &shard_idx, &shard_byte_off);
        std::vector<uint8_t> got(kTensorSize, 0);
        CUDA_OK(cudaMemcpy(got.data(),
                           (uint8_t*)bufs[shard_idx].devptr + shard_byte_off,
                           kTensorSize,
                           cudaMemcpyDeviceToHost));
        size_t mismatch = 0;
        size_t first_bad = 0;
        for (uint32_t k = 0; k < kTensorSize; ++k) {
            if (got[k] != patterns[t][k]) {
                if (mismatch == 0) first_bad = k;
                ++mismatch;
            }
        }
        if (mismatch != 0) {
            std::fprintf(stderr,
                "  tensor=%u shard=%u shard_off=%llu: "
                "%zu mismatched bytes, first @ off=%zu "
                "(got=0x%02x want=0x%02x)\n",
                t, shard_idx, (unsigned long long)shard_byte_off,
                mismatch, first_bad,
                (unsigned)got[first_bad],
                (unsigned)patterns[t][first_bad]);
            total_mismatch += (int)mismatch;
        }
    }
    if (total_mismatch != 0) {
        STEP_FAIL("byte-compare FAILED: %d total mismatches across %u IOs",
                  total_mismatch, kNumIos);
    }
    STEP_OK("byte-compare matches host write across %u IOs (%llu bytes)",
            kNumIos, (unsigned long long)kGpuFileSize);

    // Free aligned pattern buffers.
    for (auto p : patterns) ::free(p);

    // [14] tear down GPU side
    if (d_ctx != nullptr) cudaFree(d_ctx);
    bs->release_device_handle(h, /*stream=*/nullptr);
    h = nullptr;
    for (uint32_t s = 0; s < kNumShards; ++s) {
        if (bufs[s].dma != nullptr) {
            nvm_dma_unmap(bufs[s].dma);
            bufs[s].dma = nullptr;
        }
        if (bufs[s].devptr != nullptr) {
            cudaFree(bufs[s].devptr);
            bufs[s].devptr = nullptr;
        }
    }
    // R11: release_device_handle's return is async (stream-ordered host
    // callback); synchronize before the shutdowns below free the pools
    // themselves so every pending-return callback has actually run.
    CUDA_OK(cudaDeviceSynchronize());
    STEP_OK("release_device_handle + nvm_dma_unmap + cudaFree per shard");

    // [15] delete_gpu_file
    if (!bs->delete_gpu_file(gf, /*persist_now=*/true))
        STEP_FAIL("delete_gpu_file '%s'", kGpuFileName);
    STEP_OK("delete_gpu_file '%s'", kGpuFileName);

    // [16] shutdown
    if (!bs->shutdown())      STEP_FAIL("block_storage.shutdown");
    if (!storage->shutdown()) STEP_FAIL("nvme_storage.shutdown");
    STEP_OK("shutdown both layers");

    // [17] registry close
    reg.ptr->Close();
    STEP_OK("registry closed");

    std::fprintf(stderr,
        "\n=== block_storage_gpu_smoke (n_dev=%zu, num_shards=%u, "
        "ios=%u, total=%llu B): all %d steps passed ===\n",
        N_dev, kNumShards, kNumIos,
        (unsigned long long)kGpuFileSize, g_step);
    return 0;
}
