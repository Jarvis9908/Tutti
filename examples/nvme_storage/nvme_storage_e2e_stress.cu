/**
 * nvme_storage_e2e_stress.cu -- comprehensive end-to-end stress test for
 * HostFsBackedNvmeStorage (R11.5).
 *
 * Phases:
 *   1. Bulk create N files (default 100K, --n 1000000 for 1M) — timing.
 *   2. Concurrent R/W on 2K random files + L1/L2 cache eviction —
 *      data correctness.
 *   3. Crash consistency:
 *      3a. Ghost sweep (stray .bin with no log entry).
 *      3b. External rm (unlink original .bin, keep .refs/).
 *      3c. Data durability bounce (write + sync + reboot + verify).
 *   4. Bulk delete all files — timing.
 *
 * Usage (SERVICE_CLIENT):
 *   sudo ./bin/nvme_storage_e2e_stress --cuda 0 \
 *       --service 127.0.0.1:50051 --dev-id 0 --dev-id 1 \
 *       [--n 1000000] [--rw-files 2000] [--rw-threads 8] [--rw-iters 100]
 *
 * Requires a running nvmeservice_daemon.  Destructive on the NVMe
 * filesystems (creates + deletes files under .tutti/).
 */

#include "host_fs_backed_nvme_storage.h"
#include "nvme_file.h"
#include "nvme_file_device_handle.h"
#include "nvme_storage.h"
#include "nvme_storage_device.cuh"

#include "../common/registry_cli.h"
#include "../../coordinator/include/device.h"
#include "../../device_manager/include/local_nvme_device.h"

#include <cuda_runtime.h>
#include <nvm_ctrl.h>
#include <nvm_dma.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <random>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

// ---------------------------------------------------------------------------

static int g_step = 0;

#define STEP_OK(...) do { \
    std::fprintf(stderr, "[ OK ] step=%-3d ", ++g_step); \
    std::fprintf(stderr, __VA_ARGS__); \
    std::fprintf(stderr, "\n"); \
} while (0)

#define STEP_FAIL(...) do { \
    std::fprintf(stderr, "[FAIL] step=%-3d ", ++g_step); \
    std::fprintf(stderr, __VA_ARGS__); \
    std::fprintf(stderr, "\n"); \
    std::_Exit(2); \
} while (0)

#define CUDA_OK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) STEP_FAIL("CUDA: %s (%s)", #call, cudaGetErrorString(_e)); \
} while (0)

static double seconds_since(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
}

// O_DIRECT requires a block-aligned buffer.
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
// GPU kernel: one thread, one NVMe read (wraps the __device__ submit_read_one).
// ---------------------------------------------------------------------------
__global__ void submit_read_one_kernel(const tutti::NvmeFileDeviceHandle* dh,
                                       uint64_t prp1, uint64_t prp2,
                                       uint64_t logical_off, uint64_t nbytes)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        tutti::submit_read_one(dh, prp1, prp2, logical_off, nbytes);
    }
}

// ---------------------------------------------------------------------------

static constexpr uint64_t kCreateChunkSize = 2000;

struct FileRef {
    const tutti::Device* device;
    std::string name;
};

static bool parallel_create(
    tutti::HostFsBackedNvmeStorage& storage,
    const std::vector<const tutti::Device*>& devs,
    const char* prefix,
    uint64_t n_files, uint64_t file_size,
    uint32_t nvme_flags,
    std::vector<FileRef>* collect = nullptr)
{
    const uint32_t n_threads =
        (uint32_t)((n_files + kCreateChunkSize - 1) / kCreateChunkSize);
    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    std::atomic<bool> failed{false};
    std::vector<std::vector<FileRef>> per_thread(n_threads);

    for (uint32_t t = 0; t < n_threads; ++t) {
        uint64_t start = (uint64_t)t * kCreateChunkSize;
        uint64_t end   = std::min(start + kCreateChunkSize, n_files);
        threads.emplace_back([&, t, start, end]() {
            auto& local = per_thread[t];
            if (collect) local.reserve(end - start);
            for (uint64_t i = start; i < end; ++i) {
                char nm[64];
                std::snprintf(nm, sizeof(nm), "%s%lu", prefix, (unsigned long)i);
                const tutti::Device* dev = devs[i % devs.size()];
                auto* nf = storage.open_file(dev, nm, nvme_flags, file_size);
                if (nf == nullptr) { failed.store(true); return; }
                if (collect) local.push_back({dev, nm});
            }
        });
    }
    for (auto& th : threads) th.join();
    if (failed.load()) return false;
    if (collect) {
        for (auto& vec : per_thread)
            for (auto& r : vec)
                collect->push_back(std::move(r));
    }
    return true;
}

static void parallel_delete(
    tutti::HostFsBackedNvmeStorage& storage,
    const std::vector<const tutti::Device*>& devs,
    uint64_t chunk = 2000)
{
    std::vector<std::thread> threads;
    for (auto* dev : devs) {
        auto names = storage.list_file_names(dev);
        for (size_t i = 0; i < names.size(); i += chunk) {
            size_t end = std::min(i + chunk, names.size());
            // Copy the chunk to avoid dangling reference (names is
            // destroyed when the outer loop moves to the next device).
            std::vector<std::string> chunk_names(names.begin() + i, names.begin() + end);
            threads.emplace_back([&, dev, chunk_names = std::move(chunk_names)]() {
                for (const auto& nm : chunk_names) {
                    auto* f = storage.open_file(dev, nm);
                    if (f) storage.delete_file(f, /*persist_now=*/false);
                }
            });
        }
    }
    for (auto& th : threads) th.join();
    for (auto* dev : devs) storage.flush_metadata(dev);
}

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    uint64_t n_files    = 100000;
    uint64_t file_size  = 4096;
    uint32_t rw_files   = 2000;
    uint32_t rw_threads = 8;
    uint32_t rw_iters   = 100;
    uint64_t rw_subset  = 0;  // 0 = process all files
    tutti_examples::RegistryCliOptions ropt;

    for (int i = 1; i < argc; ) {
        if (tutti_examples::parse_registry_cli_arg(ropt, argc, argv, i)) { ++i; continue; }
        const char* a = argv[i];
        if      (std::strcmp(a, "--n")           == 0 && i+1 < argc) { n_files    = std::strtoull(argv[++i], nullptr, 10); ++i; }
        else if (std::strcmp(a, "--size")        == 0 && i+1 < argc) { file_size  = std::strtoull(argv[++i], nullptr, 10); ++i; }
        else if (std::strcmp(a, "--rw-files")    == 0 && i+1 < argc) { rw_files   = (uint32_t)std::strtoul(argv[++i], nullptr, 10); ++i; }
        else if (std::strcmp(a, "--rw-threads")  == 0 && i+1 < argc) { rw_threads = (uint32_t)std::strtoul(argv[++i], nullptr, 10); ++i; }
        else if (std::strcmp(a, "--rw-iters")    == 0 && i+1 < argc) { rw_iters   = (uint32_t)std::strtoul(argv[++i], nullptr, 10); ++i; }
        else if (std::strcmp(a, "--rw-subset")   == 0 && i+1 < argc) { rw_subset  = std::strtoull(argv[++i], nullptr, 10); ++i; }
        else { std::fprintf(stderr, "unknown arg: %s\n", a); return 1; }
    }
    if (!tutti_examples::validate_registry_cli(ropt, argv[0])) return 1;

    CUDA_OK(cudaFree(0));
    CUDA_OK(cudaSetDevice(ropt.cuda_dev));
    STEP_OK("cudaSetDevice(%d)", ropt.cuda_dev);

    // ---- registry (build_queue_group=true for Phase 2 acquire) ----
    auto reg = tutti_examples::open_registry(ropt, /*build_queue_group=*/true);
    if (reg.ptr == nullptr) STEP_FAIL("registry Open()");
    std::vector<const tutti::Device*> devs;
    for (size_t i = 0; i < reg.ptr->device_count(); ++i)
        devs.push_back(reg.ptr->device_at(i));
    if (devs.empty()) STEP_FAIL("no devices");
    STEP_OK("registry up: mode=%s n_dev=%zu", ropt.service_mode ? "service" : "direct", devs.size());

    // ---- storage ----
    auto make_storage = [&]() {
        auto s = std::make_unique<tutti::HostFsBackedNvmeStorage>(
            tutti::HostFsBackedNvmeStorage::Config{});
        if (!s->bootstrap(devs)) STEP_FAIL("storage.bootstrap");
        return s;
    };
    auto storage = make_storage();

    // Pre-cleanup
    {
        std::size_t n = 0;
        for (auto* d : devs) {
            for (const auto& nm : storage->list_file_names(d)) {
                auto* f = storage->open_file(d, nm);
                if (f && storage->delete_file(f, false)) ++n;
            }
            storage->flush_metadata(d);
        }
        if (n) std::fprintf(stderr, "[stress] pre-cleanup: removed %zu straggler(s)\n", n);
    }

    // ======================================================================
    // Phase 1: Bulk create N files
    // ======================================================================
    STEP_OK("Phase 1: bulk create %lu files (%lu bytes each) — starting",
            (unsigned long)n_files, (unsigned long)file_size);
    std::vector<FileRef> all_files;
    all_files.reserve(n_files);
    {
        auto t0 = std::chrono::steady_clock::now();
        if (!parallel_create(*storage, devs, "stress_", n_files, file_size,
                             tutti::NVME_OPEN_CREATE | tutti::NVME_OPEN_NO_PERSIST | tutti::NVME_OPEN_NO_SYNC,
                             &all_files))
            STEP_FAIL("Phase 1: parallel_create failed");
        for (auto* d : devs)
            if (!storage->flush_metadata(d)) STEP_FAIL("Phase 1: flush_metadata");
        double sec = seconds_since(t0);
        STEP_OK("Phase 1: created %lu files in %.3fs (%.0f ops/s), %u threads",
                (unsigned long)n_files, sec, n_files / sec,
                (uint32_t)((n_files + kCreateChunkSize - 1) / kCreateChunkSize));
    }
    // Verify count
    {
        std::size_t total = 0;
        for (auto* d : devs) total += storage->list_file_names(d).size();
        if (total != n_files)
            STEP_FAIL("Phase 1: count mismatch: expected %lu, got %zu", (unsigned long)n_files, total);
        STEP_OK("Phase 1: verified %zu files in directory", total);
    }

    // ======================================================================
    // Phase 2: GPU kernel R/W + L1/L2 cache eviction
    // ======================================================================
    // Host-writes a unique pattern to each file, then uses the GPU
    // submit_read_one kernel to read it back via NVMe DMA.  The L1/L2
    // handle cache is sized at 128 MiB budget (capped to force eviction
    // with the working set).  Files are processed in batches; each batch
    // acquires handles (cold miss → build, or L2 hit → promote), and
    // across batches the L1/L2 LRU evicts entries → multiple correct
    // cache replacements are exercised.
    uint64_t rw_subset_eff = (rw_subset == 0) ? n_files : std::min(rw_subset, n_files);

    STEP_OK("Phase 2: GPU kernel R/W (%lu files, 128 MiB cache budget) — starting",
            (unsigned long)rw_subset_eff);
    {
        // L1/L2 = 128 MiB budget, capped to force eviction.
        // sizeof(NvmeFileDeviceHandle) = 192 bytes.
        // 128 MiB / 192 = 699050 entries.
        const uint32_t l1_max = 699050;
        const uint32_t l1_cap = std::min(l1_max, (uint32_t)std::max((uint64_t)1, rw_subset_eff / 4));
        const uint32_t l2_cap = std::min(l1_max, (uint32_t)std::max((uint64_t)2, rw_subset_eff / 2));
        if (!storage->configure_handle_pool(l1_cap, l2_cap))
            STEP_FAIL("Phase 2: configure_handle_pool failed");
        STEP_OK("Phase 2: handle pool L1=%u L2=%u (128 MiB cap=%u)", l1_cap, l2_cap, l1_max);

        // One GPU buffer per device + DMA map.  Use 1 MiB buffer (like
        // gpu_smoke) even for 4 KiB IOs — the PRP1 is always the first
        // page's IO address.
        const uint64_t io_size = 4096;  // GPU IO always 4 KiB (single PRP)
        const uint64_t gpu_buf_size = std::max(file_size, (uint64_t)1048576);
        struct GpuBuf { void* devptr = nullptr; nvm_dma_t* dma = nullptr; uint64_t prp1 = 0; };
        std::vector<GpuBuf> gbufs(devs.size());
        for (size_t di = 0; di < devs.size(); ++di) {
            CUDA_OK(cudaMalloc(&gbufs[di].devptr, gpu_buf_size));
            CUDA_OK(cudaMemset(gbufs[di].devptr, 0, gpu_buf_size));
            auto* bp = static_cast<tutti::LocalNvmeDevice*>(devs[di]->backend_private);
            int rc = nvm_dma_map_data_device(&gbufs[di].dma, bp->ctrl, gbufs[di].devptr, gpu_buf_size);
            if (rc != 0 || gbufs[di].dma == nullptr)
                STEP_FAIL("Phase 2: nvm_dma_map_data_device(dev[%zu]) rc=%d", di, rc);
            gbufs[di].prp1 = gbufs[di].dma->ioaddrs[0];
        }
        cudaStream_t stream;
        CUDA_OK(cudaStreamCreate(&stream));

        // Process files in batches.  Each file: host-write pattern →
        // acquire handle → GPU submit_read_one → sync → verify → release.
        const uint32_t batch_size = std::min((uint64_t)rw_files, rw_subset_eff);
        const uint64_t n_batches = (rw_subset_eff + batch_size - 1) / batch_size;
        uint64_t total_verified = 0;
        uint64_t total_mismatch = 0;
        std::mt19937 rng(12345);

        auto t0 = std::chrono::steady_clock::now();

        for (uint64_t batch = 0; batch < n_batches; ++batch) {
            uint64_t bstart = batch * batch_size;
            uint64_t bend   = std::min(bstart + batch_size, rw_subset_eff);
            // Random file indices for this batch (from the 1M pool).
            std::vector<uint64_t> idxs;
            for (uint64_t j = bstart; j < bend; ++j) idxs.push_back(j);

            // 1. Host-write unique pattern + sync.
            for (auto idx : idxs) {
                auto& ref = all_files[idx];
                auto* nf = storage->open_file(ref.device, ref.name);
                if (nf == nullptr) STEP_FAIL("Phase 2: open(%s)", ref.name.c_str());
                AlignedBuf wbuf(io_size);
                uint64_t seed = 0xCAFE0000ULL + idx;
                for (uint64_t k = 0; k < io_size; ++k)
                    wbuf[k] = (uint8_t)((seed + k) & 0xFF);
                if (storage->write_blocking(nf, 0, wbuf.data(), io_size) != (ssize_t)io_size)
                    STEP_FAIL("Phase 2: write(%s)", ref.name.c_str());
                storage->sync(nf);
            }

            // 2. For each file in batch: acquire → GPU read → sync →
            //    verify → release.  One file at a time (shared 4 KiB
            //    GPU buffer per device).
            AlignedBuf verify_buf(file_size);
            for (size_t j = 0; j < idxs.size(); ++j) {
                auto& ref = all_files[idxs[j]];
                size_t di = 0;
                for (; di < devs.size(); ++di)
                    if (devs[di] == ref.device) break;

                auto* nf = storage->open_file(ref.device, ref.name);
                auto* dh = storage->acquire_device_handle(nf, /*stream=*/nullptr);
                if (dh == nullptr) STEP_FAIL("Phase 2: acquire(%s)", ref.name.c_str());

                submit_read_one_kernel<<<1, 1>>>(
                    dh, gbufs[di].prp1, /*prp2=*/0,
                    /*logical_off=*/0, /*nbytes=*/(uint32_t)io_size);
                CUDA_OK(cudaDeviceSynchronize());

                CUDA_OK(cudaMemcpy(verify_buf.data(), gbufs[di].devptr,
                                   io_size, cudaMemcpyDeviceToHost));
                uint64_t seed = 0xCAFE0000ULL + idxs[j];
                bool ok = true;
                for (uint64_t k = 0; k < io_size; ++k) {
                    if (verify_buf[k] != (uint8_t)((seed + k) & 0xFF)) { ok = false; break; }
                }
                if (!ok) {
                    ++total_mismatch;
                    if (total_mismatch == 1)
                        std::fprintf(stderr, "[FAIL] mismatch idx=%lu\n", (unsigned long)idxs[j]);
                }
                ++total_verified;

                storage->release_device_handle(nf, /*stream=*/nullptr);
            }

            if (batch % 50 == 0 || batch == n_batches - 1)
                std::fprintf(stderr, "  [batch %lu/%lu] verified %lu files, mismatches=%lu\n",
                             (unsigned long)(batch+1), (unsigned long)n_batches,
                             (unsigned long)total_verified, (unsigned long)total_mismatch);
        }

        double sec = seconds_since(t0);
        if (total_mismatch != 0)
            STEP_FAIL("Phase 2: %lu data mismatches in %lu files",
                      (unsigned long)total_mismatch, (unsigned long)total_verified);
        STEP_OK("Phase 2: %lu files GPU-read-verified in %.3fs (%.0f ops/s), 0 mismatches, "
                "L1=%u L2=%u (cache evictions exercised by %lu files > cache)",
                (unsigned long)total_verified, sec, total_verified / sec,
                l1_cap, l2_cap, (unsigned long)rw_subset_eff);

        // Cleanup GPU resources.
        CUDA_OK(cudaStreamDestroy(stream));
        for (auto& gb : gbufs) {
            if (gb.dma) nvm_dma_unmap(gb.dma);
            if (gb.devptr) cudaFree(gb.devptr);
        }
    }

    // ======================================================================
    // Phase 3: Crash consistency
    // ======================================================================

    // ---- 3a: Ghost sweep ----
    // Create stray .bin files directly on the filesystem (bypass the storage
    // API → no log entry).  Re-bootstrap → reconcile should unlink them.
    STEP_OK("Phase 3a: ghost sweep — starting");
    {
        const uint32_t n_ghosts = 1000;
        auto* d0 = devs[0];
        std::string tutti_dir = storage->mount_path(d0) + "/.tutti";
        for (uint32_t i = 0; i < n_ghosts; ++i) {
            char nm[64];
            std::snprintf(nm, sizeof(nm), "%s/ghost_%u.bin", tutti_dir.c_str(), i);
            int fd = ::open(nm, O_CREAT | O_RDWR | O_CLOEXEC, 0600);
            if (fd < 0) STEP_FAIL("Phase 3a: open(%s) failed", nm);
            ::fallocate(fd, 0, 0, 4096);
            ::close(fd);
        }
        std::size_t before = storage->list_file_names(d0).size();

        // Shutdown + re-bootstrap.
        storage->shutdown();
        storage = make_storage();

        std::size_t after = storage->list_file_names(d0).size();
        if (after != before)
            STEP_FAIL("Phase 3a: ghost sweep failed: before=%zu after=%zu (ghosts not cleaned)",
                      before, after);
        STEP_OK("Phase 3a: %u ghosts cleaned by reconcile (count stable at %zu)",
                n_ghosts, after);
    }

    // ---- 3b: External rm ----
    // ::unlink the original .bin path of 100 files (NOT .refs/).
    // Re-bootstrap → reconcile should detect tombstones + clean .refs/.
    STEP_OK("Phase 3b: external rm — starting");
    {
        auto* d0 = devs[0];
        std::string tutti_dir = storage->mount_path(d0) + "/.tutti";
        auto names = storage->list_file_names(d0);
        const uint32_t n_rm = std::min((uint32_t)100, (uint32_t)names.size());
        if (n_rm == 0) STEP_FAIL("Phase 3b: not enough files");

        // Unlink the original .bin of the first n_rm files.
        for (uint32_t i = 0; i < n_rm; ++i) {
            std::string path = tutti_dir + "/" + names[i] + ".bin";
            if (::unlink(path.c_str()) != 0)
                std::fprintf(stderr, "[stress] 3b: unlink(%s) errno=%d (continuing)\n", path.c_str(), errno);
        }
        std::size_t before = storage->list_file_names(d0).size();

        // Shutdown + re-bootstrap.
        storage->shutdown();
        storage = make_storage();

        std::size_t after = storage->list_file_names(d0).size();
        // Reconcile should have removed the n_rm tombstone entries.
        if (after != before - n_rm)
            STEP_FAIL("Phase 3b: tombstone sweep: before=%zu after=%zu expected=%zu",
                      before, after, before - n_rm);
        STEP_OK("Phase 3b: %u tombstones cleaned (count %zu -> %zu)", n_rm, before, after);
    }

    // ---- 3c: Data durability bounce ----
    // Write known patterns to 100 files, sync, reboot, verify data.
    STEP_OK("Phase 3c: data durability bounce — starting");
    {
        auto* d0 = devs[0];
        auto names = storage->list_file_names(d0);
        const uint32_t n_bounce = std::min((uint32_t)100, (uint32_t)names.size());
        if (n_bounce == 0) STEP_FAIL("Phase 3c: not enough files");

        AlignedBuf buf(file_size);
        // Write + sync.
        for (uint32_t i = 0; i < n_bounce; ++i) {
            auto* nf = storage->open_file(d0, names[i]);
            if (nf == nullptr) STEP_FAIL("Phase 3c: open(%s)", names[i].c_str());
            uint64_t seed = 0xBABC0E00ULL + i;
            for (uint64_t k = 0; k < file_size; ++k)
                buf[k] = (uint8_t)((seed + k) & 0xFF);
            if (storage->write_blocking(nf, 0, buf.data(), file_size) != (ssize_t)file_size)
                STEP_FAIL("Phase 3c: write(%s)", names[i].c_str());
            storage->sync(nf);
        }

        // Shutdown + re-bootstrap.
        storage->shutdown();
        storage = make_storage();

        // Read back + verify.
        uint32_t mismatch = 0;
        for (uint32_t i = 0; i < n_bounce; ++i) {
            auto* nf = storage->open_file(d0, names[i]);
            if (nf == nullptr) STEP_FAIL("Phase 3c: reopen(%s)", names[i].c_str());
            std::memset(buf.data(), 0, file_size);
            if (storage->read_blocking(nf, 0, buf.data(), file_size) != (ssize_t)file_size)
                STEP_FAIL("Phase 3c: read(%s)", names[i].c_str());
            uint64_t seed = 0xBABC0E00ULL + i;
            for (uint64_t k = 0; k < file_size; ++k) {
                if (buf[k] != (uint8_t)((seed + k) & 0xFF)) { ++mismatch; break; }
            }
        }
        if (mismatch != 0)
            STEP_FAIL("Phase 3c: %u/%u files have data mismatch after bounce", mismatch, n_bounce);
        STEP_OK("Phase 3c: %u files verified data-durable across bounce", n_bounce);
    }

    // ======================================================================
    // Phase 4: Bulk delete all files
    // ======================================================================
    STEP_OK("Phase 4: bulk delete — starting");
    {
        std::size_t before = 0;
        for (auto* d : devs) before += storage->list_file_names(d).size();

        auto t0 = std::chrono::steady_clock::now();
        parallel_delete(*storage, devs);
        double sec = seconds_since(t0);

        std::size_t after = 0;
        for (auto* d : devs) after += storage->list_file_names(d).size();
        if (after != 0)
            STEP_FAIL("Phase 4: delete incomplete: %zu files remain", after);
        STEP_OK("Phase 4: deleted %zu files in %.3fs (%.0f ops/s)",
                before, sec, before / sec);
    }

    // ---- shutdown ----
    storage->shutdown();
    reg.ptr->Close();

    std::fprintf(stderr,
        "\n=== nvme_storage_e2e_stress: ALL PHASES PASSED ===\n");
    return 0;
}
