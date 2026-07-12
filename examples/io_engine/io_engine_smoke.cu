/**
 * io_engine_smoke.cu -- R8.2 multi-round end-to-end byte-verified smoke.
 *
 * Spec smoke for R8 (doc/refactor/R8_io_engine_plan.md §4): batch IO
 * through the GPU kernel against a 2-stripe GpuFile, byte-compared
 * against ground truth in BOTH directions across multiple rounds to
 * exercise stream/queue reuse and the kernel-write -> host-read
 * coherency path.
 *
 * Layout:
 *   GpuFile      : num_shards=2, layers=4, tensor_size=128 KiB
 *                  (per-shard size = 4 * 128 KiB = 512 KiB,
 *                   total file size = 1 MiB).
 *   GPU tensor   : 1 MiB device region, granularity=128 KiB.
 *                  R7 splits it into 8 sub-slices, each one IO of
 *                  128 KiB (32 host pages -> PRP-list path active).
 *   Submit shape : 8 NvmeBatchEntries.
 *                  fd_idx   = prp_idx % 2     -> shards 0,1,0,1,0,1,0,1
 *                  shard_off= (prp_idx/2)*128 -> 0,0,128,128,256,256,...
 *
 * Rounds (each reuses the same buffers + stream + d_qps + IoSliceTable
 * + d_shards_dev + LocalNvmeIoEngine scratch to exercise re-use across
 * multiple submits):
 *   round 0 READ          host pwrite seed-r0 to shards
 *                         -> engine.submit_batch(read)
 *                         -> verify device tensor == seed-r0
 *   round 1 WRITE         seed-r1 -> device tensor (cudaMemcpy)
 *                         -> engine.submit_batch(write)
 *                         -> drop host page cache for shard fds
 *                         -> host pread -> verify shards == seed-r1
 *   round 2 READ-after-W  shards still hold seed-r1 from round 1
 *                         -> cudaMemset(0xCD) wipes device tensor so
 *                            "kernel did nothing" produces a
 *                            recognisable mismatch
 *                         -> engine.submit_batch(read)
 *                         -> verify device tensor == seed-r1
 *                         (this confirms round-1 writes actually
 *                          persisted on the NVMe controller)
 *   round 3 WRITE         seed-r3 -> device tensor
 *                         -> engine.submit_batch(write)
 *                         -> host pread (O_DIRECT); verify
 *
 * Cache coherency note for the kernel-write -> host-pread path:
 * the kernel's NVMe DMA writes bypass the host's ext4 page cache.
 * R11.5: read_blocking uses O_DIRECT, which also bypasses the page
 * cache, so host pread always fetches fresh data from disk -- no
 * posix_fadvise cache-drop is needed.  This is a smoke-side concern
 * only -- production hot path is GPU NVMe DMA (R5b).
 *
 * Test contract:
 *   - sole owner of every NVMe (NVMeService daemon MUST NOT be running)
 *   - DESTRUCTIVE on unformatted disks (mkfs.ext4 -F via nvme_storage)
 *   - exactly 2 PCI BDFs required (kNumShards = 2)
 *
 * Usage:
 *   sudo ./io_engine_smoke --cuda 0 --cap 32 \
 *        0000:4b:00.0 0000:57:00.0
 */

// io_engine
#include "host_batch_builder.h"
#include "launch_batch.h"
#include "nvme_batch.h"
#include "io_engine.h"
#include "local_nvme/local_nvme_io_engine.h"

// memory (R7)
#include "host_device_memory_subsystem.h"

// block_storage (R6 + R8.1 d_shards_dev extension)
#include "host_fs_backed_block_storage.h"
#include "block_storage.h"

// nvme_storage (R5)
#include "host_fs_backed_nvme_storage.h"
#include "nvme_storage.h"

// device_manager + runtime
#include "../common/registry_cli.h"
#include "../../coordinator/include/device.h"

#include <cuda_runtime.h>
#include <dirent.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <memory>
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

void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage:\n"
        "  IN_PROCESS: %s [--cuda N] [--cap N] <PCI_BDF> <PCI_BDF>\n"
        "  SERVICE_CLIENT: %s [--cuda N] --service <endpoint> --dev-id N --dev-id M\n"
        "  e.g.: %s --cuda 0 --cap 32 0000:4b:00.0 0000:57:00.0\n"
        "        %s --cuda 0 --service 127.0.0.1:50051 --dev-id 0 --dev-id 1\n"
        "DESTRUCTIVE on unformatted disks (mkfs.ext4 -F via nvme_storage).\n"
        "Needs exactly 2 devices (num_shards = 2).\n",
        prog, prog, prog, prog);
}

void prime_cuda(int cuda_dev) {
    cudaError_t cerr = cudaFree(0);
    if (cerr != cudaSuccess && cerr != cudaErrorInvalidValue) {
        STEP_FAIL("cuda driver prime failed: %s", cudaGetErrorString(cerr));
    }
    (void)cudaGetLastError();
    CUDA_OK(cudaSetDevice(cuda_dev));
    STEP_OK("cudaSetDevice(%d)", cuda_dev);
}

// ---------------------------------------------------------------------------
// Layout knobs (must match the docstring above).
// ---------------------------------------------------------------------------
constexpr uint32_t kNumShards    = 2;
constexpr uint32_t kLayers       = 4;
constexpr uint32_t kTensorSize   = 128u * 1024;        // 128 KiB
constexpr uint64_t kGpuFileSize  =
    (uint64_t)kNumShards * kLayers * kTensorSize;      // 1 MiB

constexpr std::size_t kRegionSize  = 1u << 20;         // 1 MiB GPU tensor
constexpr std::size_t kGranularity = 128u * 1024;      // == kTensorSize
constexpr std::size_t kNumSubSlices = kRegionSize / kGranularity;  // 8

constexpr const char* kGpuFileName = "io_engine_smoke_E2E";
constexpr const char* kPrefix      = "io_engine_smoke_";

// Distinct mixing per (tensor, round); avoids accidental aliasing
// across rounds so a "leftover from round N-1" failure mode is
// detectable instead of silently passing.
uint64_t pattern_seed_r(uint32_t tensor_idx, uint32_t round) {
    return (((uint64_t)round + 1u) * 0x9E3779B97F4A7C15ULL) ^
           (((uint64_t)tensor_idx + 1u) * 0xC2B2AE3D27D4EB4FULL) ^
           0xC0FFEE0B5BAD0B5BULL;
}

void fill_pattern(uint8_t* buf, uint32_t nbytes, uint64_t seed) {
    for (uint32_t i = 0; i < nbytes; ++i) {
        buf[i] = (uint8_t)((seed * 0xC2B2AE3D27D4EB4FULL + i) & 0xFFu);
    }
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

void wipe_stragglers(tutti::IBlockStorage& bs) {
    std::size_t n = 0;
    for (const auto& nm : bs.list_gpu_file_names()) {
        // Match our own prefix OR kvs_* files left behind by
        // kv_cache_e2e_stress.  The kvs_* sweep is important: if a
        // prior stress run left millions of files on disk, the
        // reconcile/syncfs inside bootstrap can exhaust resources
        // (fd table, memory) or take minutes, and the resulting
        // std::system_error(EINVAL) from deep inside the mutex layer
        // is not actionable.  Removing them here keeps io_engine_smoke
        // self-contained regardless of what ran before.
        if (nm.rfind(kPrefix, 0) != 0 && nm.rfind("kvs_", 0) != 0) continue;
        tutti::GpuFileSpec spec{};
        spec.name = nm;
        tutti::GpuFile* gf = bs.open_gpu_file(spec);   // EXISTING: only spec.name matters
        if (gf == nullptr) continue;
        if (bs.delete_gpu_file(gf, /*persist_now=*/false)) ++n;
    }
    if (n > 0) (void)bs.flush_metadata();
    if (n > 0) {
        std::fprintf(stderr,
            "[io_engine] pre-cleanup: removed %zu straggler GpuFile(s) "
            "(prefix '%s*' or 'kvs_*')\n",
            n, kPrefix);
    }
}

// Per-round shared state.  A reference to one of these is threaded
// through every helper so each call site reads as a single line.
struct Ctx {
    tutti::HostDeviceMemorySubsystem*   mem       = nullptr;
    tutti::HostFsBackedNvmeStorage*     storage   = nullptr;
    tutti::GpuFile*                     gf        = nullptr;
    tutti::GpuFileHandle*               h         = nullptr;
    tutti::MemoryRegion*                tr        = nullptr;
    cudaStream_t                        stream    = nullptr;
    tutti::IIoEngine*                   engine    = nullptr;
};

void gen_patterns(std::vector<uint8_t*>& dst, uint32_t round) {
    for (uint32_t t = 0; t < kNumSubSlices; ++t) {
        if (::posix_memalign(reinterpret_cast<void**>(&dst[t]), 4096, kTensorSize) != 0)
            STEP_FAIL("posix_memalign(patterns[%u])", t);
        fill_pattern(dst[t], kTensorSize, pattern_seed_r(t, round));
    }
}

void host_write_shards(Ctx& c,
                       const std::vector<uint8_t*>& patterns,
                       uint32_t round) {
    for (uint32_t t = 0; t < kNumSubSlices; ++t) {
        const uint32_t shard_idx     = t % kNumShards;
        const uint64_t shard_byte_off = (uint64_t)(t / kNumShards) * kTensorSize;
        ssize_t wr = c.storage->write_blocking(c.gf->shards[shard_idx],
                                                shard_byte_off,
                                                patterns[t],
                                                kTensorSize);
        if (wr != (ssize_t)kTensorSize)
            STEP_FAIL("[round %u host-write] write_blocking t=%u rc=%zd",
                      round, t, wr);
    }
    for (uint32_t s = 0; s < kNumShards; ++s) {
        if (!c.storage->sync(c.gf->shards[s]))
            STEP_FAIL("[round %u host-write] sync(shard=%u)", round, s);
    }
}

void host_read_shards(Ctx& c, uint8_t* out, uint32_t round) {
    std::memset(out, 0, kRegionSize);
    for (uint32_t t = 0; t < kNumSubSlices; ++t) {
        const uint32_t shard_idx     = t % kNumShards;
        const uint64_t shard_byte_off = (uint64_t)(t / kNumShards) * kTensorSize;
        ssize_t rd = c.storage->read_blocking(c.gf->shards[shard_idx],
                                               shard_byte_off,
                                               out + (uint64_t)t * kTensorSize,
                                               kTensorSize);
        if (rd != (ssize_t)kTensorSize)
            STEP_FAIL("[round %u host-read] read_blocking t=%u rc=%zd",
                      round, t, rd);
    }
}

void device_load_patterns(Ctx& c,
                          const std::vector<uint8_t*>& patterns) {
    auto* base = static_cast<uint8_t*>(c.tr->device_ptr);
    for (uint32_t t = 0; t < kNumSubSlices; ++t) {
        CUDA_OK(cudaMemcpy(base + (uint64_t)t * kTensorSize,
                           patterns[t], kTensorSize,
                           cudaMemcpyHostToDevice));
    }
}

void device_dump_to_host(Ctx& c, uint8_t* out) {
    std::memset(out, 0, kRegionSize);
    CUDA_OK(cudaMemcpy(out, c.tr->device_ptr,
                       kRegionSize, cudaMemcpyDeviceToHost));
}

// Submit one batch via IIoEngine and block until it completes.
// Single input tensor (tr) covering the whole device region; engine
// internally rebuilds NvmeBatchEntry[], H2D's, launches the kernel,
// and stream-syncs.
void engine_submit(Ctx& c, bool is_read, uint32_t round) {
    std::vector<tutti::NvmeBatchInputTensor> inputs;
    inputs.push_back({c.tr, c.h, /*file_byte_offset=*/0});
    if (!c.engine->submit_batch(inputs, is_read, c.stream)) {
        STEP_FAIL("[round %u] engine.submit_batch is_read=%d failed",
                  round, (int)is_read);
    }
}

void byte_compare_or_fail(const std::vector<uint8_t*>& expect,
                           const uint8_t* got,
                           const char* tag) {
    int total_mismatch = 0;
    int bad_slices     = 0;
    for (uint32_t t = 0; t < kNumSubSlices; ++t) {
        const uint64_t off = (uint64_t)t * kTensorSize;
        const uint8_t* g = got + off;
        const uint8_t* w = expect[t];
        std::size_t mismatch = 0;
        std::size_t first_bad = 0;
        for (uint32_t k = 0; k < kTensorSize; ++k) {
            if (g[k] != w[k]) {
                if (mismatch == 0) first_bad = k;
                ++mismatch;
            }
        }
        if (mismatch != 0) {
            ++bad_slices;
            total_mismatch += (int)mismatch;
            const uint32_t shard_idx     = t % kNumShards;
            const uint64_t shard_byte_off =
                (uint64_t)(t / kNumShards) * kTensorSize;
            std::fprintf(stderr,
                "  [%s] sub-slice t=%u (shard=%u shard_off=%llu): "
                "%zu mismatched bytes, first @ off=%zu "
                "(got=0x%02x want=0x%02x)\n",
                tag, t, shard_idx, (unsigned long long)shard_byte_off,
                mismatch, first_bad,
                (unsigned)g[first_bad], (unsigned)w[first_bad]);
        }
    }
    if (total_mismatch != 0) {
        STEP_FAIL("[%s] byte-compare FAILED: %d mismatches in %d/%zu sub-slices",
                  tag, total_mismatch, bad_slices, kNumSubSlices);
    }
}

}  // namespace

int main(int argc, char** argv) {
    tutti_examples::RegistryCliOptions ropt;

    for (int i = 1; i < argc; ) {
        if (tutti_examples::parse_registry_cli_arg(ropt, argc, argv, i)) { ++i; continue; }
        const char* a = argv[i];
        if (std::strcmp(a, "-h") == 0 || std::strcmp(a, "--help") == 0) {
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
    if (n_dev_req != kNumShards) {
        std::fprintf(stderr,
            "need exactly %u devices (got %zu) -- num_shards = %u\n",
            kNumShards, n_dev_req, kNumShards);
        usage(argv[0]); return 1;
    }

    // [1] cuda prime
    prime_cuda(ropt.cuda_dev);

    // [2] registry up -- build_queue_group=true (R5b GPU submit needs
    //     d_qps).  IN_PROCESS or SERVICE_CLIENT, mode-agnostic past
    //     this point.
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

    // [3b] Pre-bootstrap filesystem cleanup: if a prior kv_cache_e2e_stress
    //      run left millions of kvs_* shard files on disk, the
    //      reconcile_locked_() inside block_storage::bootstrap() would
    //      call syncfs() on a filesystem with millions of dirty entries,
    //      which triggers a std::system_error(EINVAL) crash deep in the
    //      mutex layer.  Unlink them directly at the filesystem level
    //      (before block_storage reads its log) so bootstrap's reconcile
    //      never sees them.
    {
        std::size_t n_unlinked = 0;
        for (const auto* d : devices) {
            std::string tutti_dir = storage->mount_path(d) + "/.tutti";
            DIR* dirp = ::opendir(tutti_dir.c_str());
            if (dirp == nullptr) continue;
            struct dirent* ent;
            while ((ent = ::readdir(dirp)) != nullptr) {
                if (std::strncmp(ent->d_name, "kvs_", 4) != 0) continue;
                std::string path = tutti_dir + "/" + ent->d_name;
                if (::unlink(path.c_str()) == 0) ++n_unlinked;
            }
            ::closedir(dirp);
        }
        if (n_unlinked > 0) {
            std::fprintf(stderr,
                "[io_engine] pre-bootstrap: unlinked %zu leftover "
                "kvs_* shard file(s) from mount point(s)\n", n_unlinked);
        }
    }

    // [4] block_storage bootstrap
    auto bs = std::make_unique<tutti::HostFsBackedBlockStorage>();
    if (!bs->bootstrap(storage.get(), devices))
        STEP_FAIL("block_storage.bootstrap");
    STEP_OK("block_storage bootstrap (entries=%zu)",
            bs->list_gpu_file_names().size());

    // [5] memory subsystem bring-up
    tutti::HostDeviceMemorySubsystem mem;
    mem.set_descriptor_format(tutti::DescriptorFormat::PRP);
    mem.bind_devices(devices);
    STEP_OK("memory subsystem ready (PRP, bind_devices=%zu)", N_dev);

    // [6] pre-cleanup
    wipe_stragglers(*bs);
    STEP_OK("pre-cleanup");

    // [7] open_gpu_file(CREATE)
    tutti::GpuFileSpec spec{};
    spec.name             = kGpuFileName;
    spec.total_size       = kGpuFileSize;
    spec.tensor_shape[0]  = kNumShards;
    spec.tensor_shape[1]  = kLayers;
    spec.tensor_shape[2]  = kTensorSize;
    spec.shard_placement  = { devices[0], devices[1] };
    tutti::GpuFile* gf = bs->open_gpu_file(
        spec, tutti::GPU_FILE_OPEN_CREATE | tutti::GPU_FILE_OPEN_EXCL);
    if (gf == nullptr) STEP_FAIL("open_gpu_file(CREATE) '%s'", kGpuFileName);
    STEP_OK("open_gpu_file(CREATE) '%s' total=%llu shards=%u tensor_size=%u layers=%u",
            kGpuFileName, (unsigned long long)kGpuFileSize,
            kNumShards, kTensorSize, kLayers);

    // [8] create the persistent stream FIRST (R11: acquire_device_handle
    //     is now async/pooled and queues its H2D fill on a stream),
    //     then acquire_device_handle + register_tensor + persistent
    //     LocalNvmeIoEngine.  All of these are reused across all 4
    //     rounds so we exercise queue/stream/scratch re-use.
    cudaStream_t stream = nullptr;
    CUDA_OK(cudaStreamCreate(&stream));

    if (!bs->configure_handle_pool(1, 1)) STEP_FAIL("configure_handle_pool");
    tutti::GpuFileHandle* h = bs->acquire_device_handle(gf, stream);
    if (h == nullptr || h->d_shards_dev == nullptr)
        STEP_FAIL("acquire_device_handle / d_shards_dev null");

    tutti::MemoryRegion* tr =
        mem.allocate_device(kRegionSize, tutti::MemoryKind::DEVICE, ropt.cuda_dev);
    if (tr == nullptr) STEP_FAIL("allocate_device(%zu)", kRegionSize);
    {
        tutti::TensorRegistrationSpec ds{};
        ds.ptr         = tr->device_ptr;
        ds.size        = tr->size;
        ds.granularity = kGranularity;
        if (mem.register_tensor(ds) != tr)
            STEP_FAIL("register_tensor(granularity=%zu)", kGranularity);
    }

    // R8.3: IIoEngine owns its own NvmeBatchEntry scratch (host +
    // device).  Sized to comfortably exceed kNumSubSlices so the
    // per-round flatten always fits.
    auto engine = std::make_unique<tutti::LocalNvmeIoEngine>(
        &mem, /*max_entries_per_batch=*/(uint32_t)kNumSubSlices * 4u);

    Ctx ctx{};
    ctx.mem       = &mem;
    ctx.storage   = storage.get();
    ctx.gf        = gf;
    ctx.h         = h;
    ctx.tr        = tr;
    ctx.stream    = stream;
    ctx.engine    = engine.get();

    STEP_OK("acquire_device_handle(stream) [async] + register_tensor(1 MiB / "
            "%zu KiB) + LocalNvmeIoEngine(max_entries=%u) staged "
            "(reused across 4 rounds)",
            kGranularity / 1024, engine->max_entries_per_batch());

    // -----------------------------------------------------------------------
    // ROUND 0: READ
    //   host pwrite seed-r0 -> kernel read -> verify device tensor.
    // -----------------------------------------------------------------------
    {
        std::vector<uint8_t*> patterns_r0(kNumSubSlices, nullptr);
        gen_patterns(patterns_r0, /*round=*/0);
        host_write_shards(ctx, patterns_r0, /*round=*/0);
        STEP_OK("[round 0 READ] host wrote %zu seed-r0 patterns + sync",
                kNumSubSlices);

        CUDA_OK(cudaMemset(tr->device_ptr, 0xAB, kRegionSize));
        engine_submit(ctx, /*is_read=*/true, /*round=*/0);
        STEP_OK("[round 0 READ] engine.submit_batch(read) ok");

        AlignedBuf got(kRegionSize);
        device_dump_to_host(ctx, got.data());
        byte_compare_or_fail(patterns_r0, got.data(), "round 0 READ");
        STEP_OK("[round 0 READ] device tensor matches seed-r0 across "
                "all %zu sub-slices (%llu B)",
                kNumSubSlices, (unsigned long long)kRegionSize);

        // -------------------------------------------------------------------
        // ROUND 1: WRITE
        //   seed-r1 -> device tensor -> kernel write -> drop cache ->
        //   host pread -> verify shards.
        // -------------------------------------------------------------------
        std::vector<uint8_t*> patterns_r1(kNumSubSlices, nullptr);
        gen_patterns(patterns_r1, /*round=*/1);
        device_load_patterns(ctx, patterns_r1);
        STEP_OK("[round 1 WRITE] device tensor loaded with seed-r1");

        engine_submit(ctx, /*is_read=*/false, /*round=*/1);
        STEP_OK("[round 1 WRITE] engine.submit_batch(write) ok");

        AlignedBuf shard_got(kRegionSize);
        host_read_shards(ctx, shard_got.data(), /*round=*/1);
        byte_compare_or_fail(patterns_r1, shard_got.data(), "round 1 WRITE");
        STEP_OK("[round 1 WRITE] shards (O_DIRECT pread) "
                "match seed-r1 across all %zu sub-slices",
                kNumSubSlices);

        // -------------------------------------------------------------------
        // ROUND 2: READ-after-WRITE
        //   shards still hold seed-r1 from round 1; kernel reads ->
        //   verify device tensor matches seed-r1.  Confirms the
        //   round-1 write actually persisted on the controller.
        // -------------------------------------------------------------------
        CUDA_OK(cudaMemset(tr->device_ptr, 0xCD, kRegionSize));
        engine_submit(ctx, /*is_read=*/true, /*round=*/2);
        STEP_OK("[round 2 READ-after-W] engine.submit_batch(read) ok");

        device_dump_to_host(ctx, got.data());
        byte_compare_or_fail(patterns_r1, got.data(), "round 2 READ-after-W");
        STEP_OK("[round 2 READ-after-W] device tensor matches "
                "round-1 written seed-r1 (confirms persistence)");

        // -------------------------------------------------------------------
        // ROUND 3: WRITE again with seed-r3
        // -------------------------------------------------------------------
        std::vector<uint8_t*> patterns_r3(kNumSubSlices, nullptr);
        gen_patterns(patterns_r3, /*round=*/3);
        device_load_patterns(ctx, patterns_r3);
        STEP_OK("[round 3 WRITE] device tensor loaded with seed-r3");

        engine_submit(ctx, /*is_read=*/false, /*round=*/3);
        STEP_OK("[round 3 WRITE] engine.submit_batch(write) ok");

        host_read_shards(ctx, shard_got.data(), /*round=*/3);
        byte_compare_or_fail(patterns_r3, shard_got.data(), "round 3 WRITE");
        STEP_OK("[round 3 WRITE] shards match seed-r3 across "
                "all %zu sub-slices", kNumSubSlices);

        // Free aligned pattern buffers.
        for (auto p : patterns_r0) ::free(p);
        for (auto p : patterns_r1) ::free(p);
        for (auto p : patterns_r3) ::free(p);
    }

    // [tear-down]  R11: release_device_handle is async -- queue it on
    //     `stream` BEFORE destroying that stream, then synchronize so
    //     the pending-return callback has actually run.
    engine.reset();
    bs->release_device_handle(h, stream);  h = nullptr;
    CUDA_OK(cudaStreamSynchronize(stream));
    cudaStreamDestroy(stream);
    if (!bs->delete_gpu_file(gf, true)) STEP_FAIL("delete_gpu_file");
    mem.free(tr);
    if (!bs->shutdown())      STEP_FAIL("block_storage.shutdown");
    if (!storage->shutdown()) STEP_FAIL("nvme_storage.shutdown");
    reg.ptr->Close();
    STEP_OK("teardown: engine destroyed, stream destroyed, "
            "gpu file deleted, registries closed");

    std::fprintf(stderr,
        "\n=== io_engine_smoke (n_dev=%zu, num_shards=%u, sub_slices=%zu, "
        "rounds=4 [R/W/R-after-W/W], total=%llu B per round): "
        "all %d steps passed ===\n",
        N_dev, kNumShards, kNumSubSlices,
        (unsigned long long)kRegionSize, g_step);
    return 0;
}
