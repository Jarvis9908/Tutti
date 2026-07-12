/**
 * phoenix_conflict_diag.cu -- diagnose why tutti (snvme.ko + libnvm)
 * cannot register GPU memory after Phoenix (phxfs.ko) has claimed a GPU.
 *
 * Background
 * ----------
 * Both modules pin GPU pages via nvidia_p2p_get_pages (loaded through
 * nvfs_nvidia_p2p_init -> __symbol_get).  Phoenix ADDITIONALLY does
 * devm_memremap_pages(..., MEMORY_DEVICE_PCI_P2PDMA) on the GPU PCIe
 * BAR4 at insmod time, claiming the BAR as ZONE_DEVICE P2P memory.
 *
 * This demo exercises the EXACT path tutti uses to register GPU memory
 * for NVMe-direct DMA: nvm_dma_map_data_device() -> snvme.ko
 * NVM_MAP_DEVICE_MEMORY ioctl -> map_gpu_memory() ->
 * nvfs_nvidia_p2p_get_pages() + nvfs_nvidia_p2p_dma_map_pages().
 *
 * Run it against a GPU that Phoenix has claimed AND against one it
 * hasn't; the contrast pinpoints the failure.
 *
 * Usage:
 *   sudo ./phoenix_conflict_diag <gpu_id> <nvme_pci_bdf>
 *   e.g.  sudo ./phoenix_conflict_diag 4 0000:08:00.0
 *         sudo ./phoenix_conflict_diag 0 0000:08:00.0
 *
 * Prerequisites:
 *   - snvme.ko loaded (lsmod | grep snvme)
 *   - Phoenix phxfs.ko loaded if testing the conflict (lsmod | grep phxfs)
 *   - /dev/snvm_control exists
 *   - No other process owns the target NVMe controller
 *
 * Non-destructive: only DMA mapping, no LBA writes.
 */

#include <cuda_runtime.h>

#include <nvm_ctrl.h>
#include <nvm_dma.h>
#include <nvm_types.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static int g_step = 0;

#define LOG_OK(fmt, ...) \
    do { ++g_step; \
         std::fprintf(stderr, "[ OK ] step=%-2d  " fmt "\n", g_step, ##__VA_ARGS__); \
    } while (0)

#define LOG_FAIL(fmt, ...) \
    do { ++g_step; \
         std::fprintf(stderr, "[FAIL] step=%-2d  " fmt "\n", g_step, ##__VA_ARGS__); \
    } while (0)

#define LOG_INFO(fmt, ...) \
    std::fprintf(stderr, "[INFO]       " fmt "\n", ##__VA_ARGS__)

static void print_gpu_info(int gpu_id) {
    int domain = 0, bus = 0, dev = 0;
    cudaDeviceGetAttribute(&domain, cudaDevAttrPciDomainId, gpu_id);
    cudaDeviceGetAttribute(&bus,   cudaDevAttrPciBusId,     gpu_id);
    cudaDeviceGetAttribute(&dev,   cudaDevAttrPciDeviceId,  gpu_id);
    char name[256] = {};
    cudaDeviceGetPCIBusId(name, sizeof(name), gpu_id);
    LOG_INFO("GPU %d: PCI %04x:%02x:%02x.0  (cudaDeviceGetPCIBusId='%s')",
             gpu_id, domain, bus, dev, name);

    // Check if /sys/class/phxfs-generic exists and which GPU it covers
    FILE* fp = std::fopen("/sys/class/phxfs-generic", "r");
    if (fp) {
        std::fclose(fp);
        LOG_INFO("  /sys/class/phxfs-generic exists (phxfs.ko loaded)");
    } else {
        LOG_INFO("  /sys/class/phxfs-generic NOT found (phxfs.ko not loaded)");
    }

    // Check if /dev/phxfs_dev* exists
    bool found_phxfs = false;
    for (int i = 0; i < 8; i++) {
        char path[64];
        std::snprintf(path, sizeof(path), "/dev/phxfs_dev%d", i);
        if (std::fopen(path, "r") != nullptr) {
            found_phxfs = true;
            LOG_INFO("  %s exists", path);
            // Try to read the PCI BDF from sysfs
            char sysfs[128];
            std::snprintf(sysfs, sizeof(sysfs),
                          "/sys/class/phxfs-generic/phxfs_dev%d/pci_bdf", i);
            FILE* bf = std::fopen(sysfs, "r");
            if (bf) {
                char bdf[64] = {};
                if (std::fgets(bdf, sizeof(bdf), bf)) {
                    // Strip trailing newline
                    char* nl = std::strchr(bdf, '\n');
                    if (nl) *nl = '\0';
                    LOG_INFO("    pci_bdf = %s", bdf);
                }
                std::fclose(bf);
            }
            break;  // only need to know it exists
        }
    }
    if (!found_phxfs) {
        LOG_INFO("  No /dev/phxfs_dev* devices found");
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr,
            "Usage: %s <gpu_id> <nvme_pci_bdf>\n"
            "  e.g.: %s 4 0000:08:00.0   (test GPU 4, the Phoenix-claimed one)\n"
            "        %s 0 0000:08:00.0   (test GPU 0, a non-Phoenix GPU)\n"
            "\n"
            "Diagnoses whether nvidia_p2p_get_pages (via snvme.ko) works\n"
            "for GPU memory registration when Phoenix phxfs.ko is loaded.\n"
            "Non-destructive (DMA mapping only, no LBA writes).\n",
            argv[0], argv[0], argv[0]);
        return 1;
    }

    const int   gpu_id     = std::atoi(argv[1]);
    const char* nvme_bdf   = argv[2];

    std::fprintf(stderr,
        "=== Phoenix Conflict Diagnostic ===\n"
        "  Target GPU:      %d\n"
        "  NVMe PCI BDF:    %s\n"
        "\n",
        gpu_id, nvme_bdf);

    // ---- Step 1: CUDA setup ----
    {
        int dev_count = 0;
        cudaError_t e = cudaGetDeviceCount(&dev_count);
        if (e != cudaSuccess) {
            LOG_FAIL("cudaGetDeviceCount: %s", cudaGetErrorString(e));
            return 2;
        }
        if (gpu_id < 0 || gpu_id >= dev_count) {
            LOG_FAIL("gpu_id %d out of range (device count=%d)", gpu_id, dev_count);
            return 2;
        }
        LOG_OK("cudaGetDeviceCount=%d, gpu %d is valid", dev_count, gpu_id);
    }

    // Prime CUDA context on the target GPU (required before nvidia_p2p_get_pages)
    {
        cudaError_t e = cudaSetDevice(gpu_id);
        if (e != cudaSuccess) {
            LOG_FAIL("cudaSetDevice(%d): %s", gpu_id, cudaGetErrorString(e));
            return 2;
        }
        e = cudaFree(0);  // prime the context
        if (e != cudaSuccess && e != cudaErrorInvalidValue) {
            LOG_FAIL("cudaFree(0) context prime: %s", cudaGetErrorString(e));
            return 2;
        }
        (void)cudaGetLastError();
        LOG_OK("CUDA context primed on GPU %d", gpu_id);
    }

    print_gpu_info(gpu_id);

    // ---- Step 2: Bring up NVMe controller via libnvm B3 ----
    nvm_ctrl_t* ctrl = nullptr;
    struct disk d;
    std::memset(&d, 0, sizeof(d));

    {
        int rc = nvm_controller_init_b3(&ctrl,
                                         "/dev/snvm_control",
                                         nvme_bdf,
                                         32,   // kernel_ioq_cap
                                         &d);
        if (rc != 0 || ctrl == nullptr) {
            LOG_FAIL("nvm_controller_init_b3('%s') rc=%d (snvme.ko loaded? "
                     "/dev/snvm_control exists? controller free?)",
                     nvme_bdf, rc);
            return 2;
        }
        if (ctrl->page_size == 0) {
            LOG_FAIL("ctrl->page_size == 0 (BAR0 mmap failed?)");
            return 2;
        }
        LOG_OK("nvm_controller_init_b3: page_size=%zu dstrd=%u max_qs=%u "
               "disk='%s'",
               ctrl->page_size, ctrl->dstrd, ctrl->max_qs, d.disk_name);
    }

    // ---- Step 3: Allocate GPU memory on the target GPU ----
    //
    // snvme.ko's NVM_MAP_DEVICE_MEMORY ioctl requires:
    //   (a) size is a multiple of GPU_PAGE_SIZE (64 KiB)
    //   (b) vaddr is 64 KiB-aligned
    // cudaMalloc only guarantees ~256 B alignment, so we over-allocate
    // and align manually (same trick as HostDeviceMemorySubsystem).
    constexpr std::size_t kGpuPageSize = 1ULL << 16;   // 64 KiB
    constexpr std::size_t kBufSize     = 1ULL << 20;   // 1 MiB
    constexpr std::size_t kAllocSize   = kBufSize + kGpuPageSize;

    void* raw_devptr = nullptr;
    void* aligned_devptr = nullptr;
    {
        cudaError_t e = cudaMalloc(&raw_devptr, kAllocSize);
        if (e != cudaSuccess) {
            LOG_FAIL("cudaMalloc(%zu B) on GPU %d: %s",
                     kAllocSize, gpu_id, cudaGetErrorString(e));
            goto cleanup_ctrl;
        }
        std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(raw_devptr);
        addr = (addr + kGpuPageSize - 1) & ~(kGpuPageSize - 1);
        aligned_devptr = reinterpret_cast<void*>(addr);

        // Verify the pointer is device memory
        cudaPointerAttributes attr{};
        e = cudaPointerGetAttributes(&attr, aligned_devptr);
        if (e != cudaSuccess) {
            LOG_FAIL("cudaPointerGetAttributes: %s", cudaGetErrorString(e));
            goto cleanup_gpu;
        }
        if (attr.type != cudaMemoryTypeDevice) {
            LOG_FAIL("ptr is NOT cudaMemoryTypeDevice (type=%d)", (int)attr.type);
            goto cleanup_gpu;
        }
        LOG_OK("cudaMalloc: raw=%p aligned=%p size=%zu (type=device, gpu=%d)",
               raw_devptr, aligned_devptr, kBufSize, attr.device);
    }

    // ---- Step 4: THE CRITICAL TEST -- nvm_dma_map_data_device ----
    //
    // This is the exact call tutti's HostDeviceMemorySubsystem::ensure_mapping_locked
    // makes.  It flows through:
    //   libnvm nvm_dma_map_data_device
    //     -> ioctl(fd, NVM_MAP_DEVICE_MEMORY, ...)
    //       -> snvme.ko map_device_memory -> map_gpu_memory
    //         -> nvfs_nvidia_p2p_get_pages(...)   <-- PIN GPU PAGES
    //         -> nvfs_nvidia_p2p_dma_map_pages(..) <-- CREATE DMA MAPPING
    //
    // If Phoenix's phxfs.ko has done devm_memremap_pages on this GPU's BAR,
    // one of these two NVIDIA P2P calls will fail here.
    {
        nvm_dma_t* dma = nullptr;
        LOG_INFO("Calling nvm_dma_map_data_device(ctrl=%p, devptr=%p, size=%zu)...",
                 (void*)ctrl, aligned_devptr, kBufSize);
        int rc = nvm_dma_map_data_device(&dma, ctrl, aligned_devptr, kBufSize);

        if (rc != 0 || dma == nullptr) {
            LOG_FAIL("nvm_dma_map_data_device FAILED: rc=%d dma=%p", rc, (void*)dma);
            LOG_INFO("");
            LOG_INFO("=== DIAGNOSIS ===");
            LOG_INFO("GPU memory registration via snvme.ko -> nvidia_p2p_get_pages");
            LOG_INFO("FAILED on GPU %d.", gpu_id);
            LOG_INFO("");
            LOG_INFO("This is the symptom of the Phoenix conflict.");
            LOG_INFO("Check dmesg for the kernel-side error:");
            LOG_INFO("  sudo dmesg | tail -50");
            LOG_INFO("");
            LOG_INFO("Compare with a non-Phoenix GPU to confirm:");
            LOG_INFO("  sudo %s <other_gpu_id> %s", argv[0], nvme_bdf);
            LOG_INFO("");
            LOG_INFO("Likely root cause: phxfs.ko's devm_memremap_pages() on GPU %d's", gpu_id);
            LOG_INFO("PCIe BAR4 has changed the BAR's page type to ZONE_DEVICE /");
            LOG_INFO("MEMORY_DEVICE_PCI_P2PDMA, which interferes with the NVIDIA");
            LOG_INFO("driver's nvidia_p2p_get_pages / nvidia_p2p_dma_map_pages.");
            goto cleanup_gpu;
        }

        LOG_OK("nvm_dma_map_data_device SUCCEEDED!");
        LOG_INFO("  dma handle:    %p", (void*)dma);
        LOG_INFO("  n_ioaddrs:     %zu", (std::size_t)dma->n_ioaddrs);
        LOG_INFO("  page_size:     %zu", (std::size_t)dma->page_size);
        if (dma->n_ioaddrs > 0) {
            LOG_INFO("  ioaddrs[0]:    0x%lx", (unsigned long)dma->ioaddrs[0]);
            LOG_INFO("  ioaddrs[last]: 0x%lx",
                     (unsigned long)dma->ioaddrs[dma->n_ioaddrs - 1]);
        }
        LOG_INFO("");
        LOG_INFO("=== RESULT ===");
        LOG_INFO("GPU %d memory CAN be registered via snvme.ko.", gpu_id);
        LOG_INFO("If Phoenix is loaded and this GPU is NOT claimed by it,");
        LOG_INFO("the conflict is GPU-specific (only the Phoenix-claimed GPU fails).");
        LOG_INFO("");

        // Clean up the DMA mapping
        nvm_dma_unmap(dma);
        LOG_OK("nvm_dma_unmap released the DMA mapping");
    }

    // ---- Cleanup ----
cleanup_gpu:
    if (raw_devptr != nullptr) {
        cudaFree(raw_devptr);
        LOG_OK("cudaFree(%p) released GPU memory", raw_devptr);
    }

cleanup_ctrl:
    if (ctrl != nullptr) {
        nvm_ctrl_free(ctrl);
        LOG_OK("nvm_ctrl_free released the controller (unbind + chrdev_remove)");
    }

    std::fprintf(stderr,
        "\n=== Diagnostic complete (GPU %d, %d steps) ===\n",
        gpu_id, g_step);
    return 0;
}
