/**
 * io_engine_host_smoke.cu -- R8.1 host-side build_nvme_batch smoke.
 *
 * Brings up the full bottom-up stack (registry -> nvme_storage ->
 * block_storage -> memory) and exercises:
 *
 *   - HostFsBackedBlockStorage::acquire_device_handle now populates
 *     GpuFileHandle::d_shards_dev (R8.1's R6 micro-extension).
 *     We cudaMemcpy the device array back and assert it matches
 *     d_shards_host element-by-element.
 *
 *   - tutti::build_nvme_batch over a single (tensor, file) input
 *     produces N entries with the legacy `fill_ctx` field formula:
 *         shards      = file_handle->d_shards_dev
 *         num_shards  = file_handle->num_shards
 *         prp_entry   = &(IoSliceView[i].d_ios[k])
 *         prp_idx     = global running index across (i, k)
 *         file_offset = input.file_byte_offset
 *         is_read     = batch direction
 *
 *   - Multi-tensor case: prp_idx resets to 0 at each input tensor,
 *     mirroring legacy `fill_ctx` (which loops over cache_mappings
 *     PER TENSOR and treats `j` as a per-tensor running index).
 *
 *   - Empty / oversize / null-input failure paths exit cleanly with
 *     a stderr diagnostic.
 *
 * NO CUDA kernel launch.  NO cudaMemcpy of NvmeBatchEntry to GPU.
 * Those land in R8.2 (`io_engine_smoke` end-to-end byte-verify).
 *
 * Test contract:
 *   - sole owner of every NVMe (NVMeService daemon MUST NOT be running)
 *   - DESTRUCTIVE on unformatted disks (mkfs.ext4 -F via nvme_storage)
 *   - >= 2 PCI BDFs required (num_shards = 2)
 *
 * Usage:
 *   sudo ./io_engine_host_smoke --cuda 0 --cap 32 \
 *        0000:4b:00.0 0000:57:00.0
 */

// io_engine
#include "host_batch_builder.h"
#include "nvme_batch.h"

// memory (R7)
#include "host_device_memory_subsystem.h"

// block_storage (R6 + R8.1 d_shards_dev extension)
#include "host_fs_backed_block_storage.h"
#include "block_storage.h"
#include "gpu_file_resolve.h"

// nvme_storage (R5)
#include "host_fs_backed_nvme_storage.h"
#include "nvme_storage.h"

// device_manager + runtime
#include "../common/registry_cli.h"
#include "../../coordinator/include/device.h"

#include <cuda_runtime.h>
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
        "  IN_PROCESS: %s [--cuda N] [--cap N] <PCI_BDF> <PCI_BDF> [...]\n"
        "  SERVICE_CLIENT: %s [--cuda N] --service <endpoint> --dev-id N --dev-id M\n"
        "  e.g.: %s --cuda 0 --cap 32 0000:4b:00.0 0000:57:00.0\n"
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
    CUDA_OK(cudaSetDevice(cuda_dev));
    STEP_OK("cudaSetDevice(%d)", cuda_dev);
}

// ---------------------------------------------------------------------------
// Layout knobs
//
// 1 MiB tensor / 128 KiB granularity / 2 shards.  granularity ==
// typical MDTS so num_ios per IoSliceView == 1.  Total sub-slices
// per tensor = 1 MiB / 128 KiB = 8 -> 8 NvmeBatchEntries per input.
// ---------------------------------------------------------------------------
constexpr uint32_t kNumShards     = 2;
constexpr uint32_t kLayers        = 1;            // 1 row of tensors
constexpr uint32_t kTensorSize    = 1u << 17;     // 128 KiB
constexpr uint64_t kGpuFileSize   =
    (uint64_t)kNumShards * kLayers * kTensorSize; // 256 KiB total
constexpr std::size_t kRegionSize    = 1u << 20;     // 1 MiB GPU tensor region
constexpr std::size_t kGranularity   = 128u * 1024;  // == kTensorSize

constexpr const char* kGpuFileNameA = "io_engine_host_smoke_A";
constexpr const char* kGpuFileNameB = "io_engine_host_smoke_B";
constexpr const char* kPrefix       = "io_engine_host_smoke_";

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
            "[io_engine] pre-cleanup: removed %zu '%s*' GpuFile straggler(s)\n",
            n, kPrefix);
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
    if (n_dev_req < kNumShards) {
        std::fprintf(stderr,
            "need at least %u devices (got %zu) -- num_shards = %u\n",
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

    // [4] block_storage bootstrap
    auto bs = std::make_unique<tutti::HostFsBackedBlockStorage>();
    if (!bs->bootstrap(storage.get(), devices))
        STEP_FAIL("block_storage.bootstrap");
    STEP_OK("block_storage bootstrap (entries=%zu)",
            bs->list_gpu_file_names().size());

    // [5] memory subsystem bring-up: format + bind cluster.
    tutti::HostDeviceMemorySubsystem mem;
    mem.set_descriptor_format(tutti::DescriptorFormat::PRP);
    mem.bind_devices(devices);
    STEP_OK("memory subsystem ready (PRP, bind_devices=%zu)", N_dev);

    // [6] pre-cleanup
    wipe_stragglers(*bs);
    STEP_OK("pre-cleanup");

    // [7] open_gpu_file(CREATE) A (2 shards: devices[0], devices[1])
    tutti::GpuFileSpec specA{};
    specA.name             = kGpuFileNameA;
    specA.total_size       = kGpuFileSize;
    specA.tensor_shape[0]  = kNumShards;
    specA.tensor_shape[1]  = kLayers;
    specA.tensor_shape[2]  = kTensorSize;
    specA.shard_placement  = { devices[0], devices[1] };
    tutti::GpuFile* gfA = bs->open_gpu_file(
        specA, tutti::GPU_FILE_OPEN_CREATE | tutti::GPU_FILE_OPEN_EXCL);
    if (gfA == nullptr) STEP_FAIL("open_gpu_file(CREATE) A");
    STEP_OK("open_gpu_file(CREATE) A '%s' shards=%u tensor_size=%u",
            kGpuFileNameA, kNumShards, kTensorSize);

    // [8] configure the async handle pool (R11) + acquire_device_handle
    //     on A; assert d_shards_dev is populated.  Stream = default
    //     (nullptr/0) -- this smoke is host-side only, no kernels.
    if (!bs->configure_handle_pool(2, 2)) STEP_FAIL("configure_handle_pool");
    tutti::GpuFileHandle* hA = bs->acquire_device_handle(gfA, /*stream=*/nullptr);
    if (hA == nullptr)            STEP_FAIL("acquire_device_handle(A)");
    if (hA->num_shards != kNumShards)
        STEP_FAIL("hA->num_shards=%u (want %u)", hA->num_shards, kNumShards);
    if (hA->d_shards_dev == nullptr)
        STEP_FAIL("hA->d_shards_dev == nullptr (R8.1 must populate)");
    {
        // Verify the array itself is GPU memory.
        cudaPointerAttributes attr{};
        cudaError_t cerr = cudaPointerGetAttributes(&attr, hA->d_shards_dev);
        if (cerr != cudaSuccess)
            STEP_FAIL("cudaPointerGetAttributes(d_shards_dev) failed: %s",
                      cudaGetErrorString(cerr));
        if (attr.type != cudaMemoryTypeDevice)
            STEP_FAIL("d_shards_dev not GPU memory (type=%d)", (int)attr.type);

        // Copy back and compare element-by-element to d_shards_host.
        std::vector<tutti::NvmeFileDeviceHandle*> roundtrip(kNumShards);
        CUDA_OK(cudaMemcpy(roundtrip.data(),
                           hA->d_shards_dev,
                           sizeof(tutti::NvmeFileDeviceHandle*) * kNumShards,
                           cudaMemcpyDeviceToHost));
        for (uint32_t s = 0; s < kNumShards; ++s) {
            if (roundtrip[s] != hA->d_shards_host[s])
                STEP_FAIL("d_shards_dev[%u]=%p mismatches d_shards_host[%u]=%p",
                          s, (void*)roundtrip[s], s, (void*)hA->d_shards_host[s]);
            if (roundtrip[s] == nullptr)
                STEP_FAIL("d_shards_dev[%u] == nullptr", s);
        }
    }
    STEP_OK("acquire_device_handle(A): d_shards_dev populated + matches "
            "d_shards_host[%u]", kNumShards);

    // [9] allocate 1 MiB device tensor + register_tensor(granularity=128 KiB).
    tutti::MemoryRegion* tensor_A =
        mem.allocate_device(kRegionSize, tutti::MemoryKind::DEVICE, ropt.cuda_dev);
    if (tensor_A == nullptr) STEP_FAIL("allocate_device(1 MiB) for tensor A");
    {
        tutti::TensorRegistrationSpec ds{};
        ds.ptr         = tensor_A->device_ptr;
        ds.size        = tensor_A->size;
        ds.granularity = kGranularity;
        if (mem.register_tensor(ds) != tensor_A)
            STEP_FAIL("register_tensor(tensor A)");
    }
    STEP_OK("register_tensor A: 1 MiB / granularity=%zu KiB",
            kGranularity / 1024);

    // [10] list_io_slices: cache the handles for later cross-check.
    std::vector<tutti::IoSliceView> viewsA = mem.list_io_slices(tensor_A);
    if (viewsA.empty()) STEP_FAIL("list_io_slices(A) empty");
    const std::size_t kExpectedSubSlicesA = []{
        std::size_t total = 0;
        // each view contributes num_ios sub-slices; for our knob choices
        // we expect num_ios == 1 per view, so total == num_views == 8.
        return total;
    }();
    (void)kExpectedSubSlicesA;
    {
        std::size_t total_subs = 0;
        for (const auto& v : viewsA) total_subs += v.num_ios;
        if (total_subs != kRegionSize / kGranularity)
            STEP_FAIL("expected %zu sub-slices on A, got %zu",
                      kRegionSize / kGranularity, total_subs);
    }
    STEP_OK("list_io_slices(A): %zu views, total_sub=%zu",
            viewsA.size(), kRegionSize / kGranularity);

    // [11] build_nvme_batch: single input.
    constexpr uint64_t kBaseOffA = 0;
    {
        std::vector<tutti::NvmeBatchInputTensor> inputs;
        inputs.push_back({tensor_A, hA, kBaseOffA});

        std::vector<tutti::NvmeBatchEntry> entries(64);
        uint32_t count = 0;
        if (!tutti::build_nvme_batch(&mem, inputs,
                                     /*is_read=*/true,
                                     entries.data(),
                                     (uint32_t)entries.size(),
                                     &count))
            STEP_FAIL("build_nvme_batch(single input)");
        if (count != kRegionSize / kGranularity)
            STEP_FAIL("count=%u (want %zu)", count, kRegionSize / kGranularity);

        // Field-by-field check against legacy fill_ctx formula.
        uint32_t expected_prp_idx = 0;
        std::size_t cursor_view = 0;
        std::size_t cursor_sub  = 0;
        for (uint32_t k = 0; k < count; ++k) {
            const tutti::NvmeBatchEntry& e = entries[k];

            // shards / num_shards
            if (e.shards != hA->d_shards_dev)
                STEP_FAIL("entries[%u].shards=%p != d_shards_dev=%p",
                          k, (void*)e.shards, (void*)hA->d_shards_dev);
            if (e.num_shards != kNumShards)
                STEP_FAIL("entries[%u].num_shards=%u (want %u)",
                          k, e.num_shards, kNumShards);

            // prp_entry: walk views in order, sub-slices in order.
            const tutti::IoSliceView& v = viewsA[cursor_view];
            const tutti::AddressDescriptor* expected_prp =
                v.d_ios + cursor_sub;
            if (e.prp_entry != expected_prp)
                STEP_FAIL("entries[%u].prp_entry=%p (want %p) "
                          "view=%zu sub=%zu",
                          k, (void*)e.prp_entry, (void*)expected_prp,
                          cursor_view, cursor_sub);

            // prp_idx: tensor-global running index.
            if (e.prp_idx != expected_prp_idx)
                STEP_FAIL("entries[%u].prp_idx=%u (want %u)",
                          k, e.prp_idx, expected_prp_idx);

            // file_offset: same value for every entry of this input.
            if (e.file_offset != kBaseOffA)
                STEP_FAIL("entries[%u].file_offset=%llu (want %llu)",
                          k, (unsigned long long)e.file_offset,
                          (unsigned long long)kBaseOffA);
            if (!e.is_read)
                STEP_FAIL("entries[%u].is_read=false (want true)", k);

            // Advance walking cursor.
            ++expected_prp_idx;
            ++cursor_sub;
            if (cursor_sub == v.num_ios) {
                cursor_sub = 0;
                ++cursor_view;
            }
        }
    }
    STEP_OK("build_nvme_batch(1 input, %zu sub-slices) "
            "field-by-field matches legacy fill_ctx formula",
            kRegionSize / kGranularity);

    // [12] Multi-tensor case: prp_idx must reset to 0 at each tensor.
    //      Also exercises non-zero file_byte_offset on the second input.
    tutti::GpuFileSpec specB{};
    specB.name             = kGpuFileNameB;
    specB.total_size       = kGpuFileSize;
    specB.tensor_shape[0]  = kNumShards;
    specB.tensor_shape[1]  = kLayers;
    specB.tensor_shape[2]  = kTensorSize;
    specB.shard_placement  = { devices[0], devices[1] };
    tutti::GpuFile* gfB = bs->open_gpu_file(
        specB, tutti::GPU_FILE_OPEN_CREATE | tutti::GPU_FILE_OPEN_EXCL);
    if (gfB == nullptr) STEP_FAIL("open_gpu_file(CREATE) B");

    tutti::GpuFileHandle* hB = bs->acquire_device_handle(gfB, /*stream=*/nullptr);
    if (hB == nullptr || hB->d_shards_dev == nullptr)
        STEP_FAIL("acquire_device_handle(B) / d_shards_dev null");

    tutti::MemoryRegion* tensor_B =
        mem.allocate_device(kRegionSize, tutti::MemoryKind::DEVICE, ropt.cuda_dev);
    if (tensor_B == nullptr) STEP_FAIL("allocate_device(B)");
    {
        tutti::TensorRegistrationSpec ds{};
        ds.ptr         = tensor_B->device_ptr;
        ds.size        = tensor_B->size;
        ds.granularity = kGranularity;
        if (mem.register_tensor(ds) != tensor_B) STEP_FAIL("register_tensor(B)");
    }
    std::vector<tutti::IoSliceView> viewsB = mem.list_io_slices(tensor_B);

    constexpr uint64_t kBaseOffB = 1ull << 30;     // arbitrary non-zero
    {
        std::vector<tutti::NvmeBatchInputTensor> inputs;
        inputs.push_back({tensor_A, hA, kBaseOffA});
        inputs.push_back({tensor_B, hB, kBaseOffB});

        std::vector<tutti::NvmeBatchEntry> entries(64);
        uint32_t count = 0;
        if (!tutti::build_nvme_batch(&mem, inputs,
                                     /*is_read=*/false,
                                     entries.data(),
                                     (uint32_t)entries.size(),
                                     &count))
            STEP_FAIL("build_nvme_batch(2 inputs)");
        const uint32_t kSubsPerInput = (uint32_t)(kRegionSize / kGranularity);
        if (count != 2 * kSubsPerInput)
            STEP_FAIL("count=%u (want %u)", count, 2 * kSubsPerInput);

        for (uint32_t k = 0; k < count; ++k) {
            const tutti::NvmeBatchEntry& e = entries[k];
            const bool      onB        = (k >= kSubsPerInput);
            const uint32_t  local_idx  = k - (onB ? kSubsPerInput : 0u);
            const auto&     views      = onB ? viewsB : viewsA;
            const tutti::GpuFileHandle* expected_h = onB ? hB : hA;
            const uint64_t  expected_off =
                onB ? kBaseOffB : kBaseOffA;

            if (e.shards != expected_h->d_shards_dev)
                STEP_FAIL("[multi] entries[%u].shards mismatch (input %s)",
                          k, onB ? "B" : "A");
            if (e.prp_idx != local_idx)
                STEP_FAIL("[multi] entries[%u].prp_idx=%u (want %u, "
                          "input %s) -- prp_idx must reset per tensor",
                          k, e.prp_idx, local_idx, onB ? "B" : "A");
            if (e.file_offset != expected_off)
                STEP_FAIL("[multi] entries[%u].file_offset=%llu (want %llu, "
                          "input %s)", k,
                          (unsigned long long)e.file_offset,
                          (unsigned long long)expected_off, onB ? "B" : "A");
            if (e.is_read)
                STEP_FAIL("[multi] entries[%u].is_read=true (want false)", k);

            // prp_entry walks views' d_ios in order.
            const tutti::IoSliceView& v =
                views[local_idx];   // num_ios == 1 by knob choice
            if (e.prp_entry != v.d_ios)
                STEP_FAIL("[multi] entries[%u].prp_entry=%p (want %p, "
                          "input %s view=%u)",
                          k, (void*)e.prp_entry, (void*)v.d_ios,
                          onB ? "B" : "A", local_idx);
        }
    }
    STEP_OK("build_nvme_batch(2 inputs, %u entries each) "
            "prp_idx resets per tensor; file_offset honoured",
            (uint32_t)(kRegionSize / kGranularity));

    // [13] Failure paths.
    {
        std::vector<tutti::NvmeBatchEntry> entries(8);
        uint32_t count = 99;

        // 13a: out_capacity too small.
        std::vector<tutti::NvmeBatchInputTensor> too_big;
        too_big.push_back({tensor_A, hA, kBaseOffA});
        bool ok = tutti::build_nvme_batch(&mem, too_big, /*is_read=*/true,
                                          entries.data(), 4, &count);
        if (ok) STEP_FAIL("[fail] expected build_nvme_batch to refuse "
                          "out_capacity=4 for 8 sub-slices");

        // 13b: null tensor_region.
        std::vector<tutti::NvmeBatchInputTensor> bad_tr;
        bad_tr.push_back({nullptr, hA, kBaseOffA});
        ok = tutti::build_nvme_batch(&mem, bad_tr, /*is_read=*/true,
                                     entries.data(),
                                     (uint32_t)entries.size(), &count);
        if (ok) STEP_FAIL("[fail] expected build_nvme_batch to refuse "
                          "null tensor_region");

        // 13c: null file_handle.
        std::vector<tutti::NvmeBatchInputTensor> bad_fh;
        bad_fh.push_back({tensor_A, nullptr, kBaseOffA});
        ok = tutti::build_nvme_batch(&mem, bad_fh, /*is_read=*/true,
                                     entries.data(),
                                     (uint32_t)entries.size(), &count);
        if (ok) STEP_FAIL("[fail] expected build_nvme_batch to refuse "
                          "null file_handle");
    }
    STEP_OK("rejection paths: out_capacity, null tensor_region, "
            "null file_handle all rejected");

    // [14] tear down everything.  R11: release_device_handle is async
    //     (stream-ordered host callback); synchronize before the
    //     shutdowns below free the pools themselves.
    bs->release_device_handle(hA, /*stream=*/nullptr);  hA = nullptr;
    bs->release_device_handle(hB, /*stream=*/nullptr);  hB = nullptr;
    CUDA_OK(cudaDeviceSynchronize());
    if (!bs->delete_gpu_file(gfA, true)) STEP_FAIL("delete_gpu_file A");
    if (!bs->delete_gpu_file(gfB, true)) STEP_FAIL("delete_gpu_file B");
    mem.free(tensor_A);
    mem.free(tensor_B);
    if (!bs->shutdown())      STEP_FAIL("block_storage.shutdown");
    if (!storage->shutdown()) STEP_FAIL("nvme_storage.shutdown");
    reg.ptr->Close();
    STEP_OK("teardown: handles released, gpu files deleted, regions freed, "
            "registries closed");

    std::fprintf(stderr,
        "\n=== io_engine_host_smoke (n_dev=%zu, num_shards=%u): all %d steps passed ===\n",
        N_dev, kNumShards, g_step);
    return 0;
}
