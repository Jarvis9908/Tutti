# Linux Kernel Portability Design

> **Status**: Design proposal. Not yet implemented. This document
> addresses the requirement to adapt Tutti to different Linux kernel
> versions and to isolate kernel-facing logic from upper-layer APIs.

## 1. Problem Statement

Tutti's `local_nvme` backend depends on a **modified Linux NVMe kernel
module** (`snvme`) that patches the stock `nvme` driver to expose
user-space queue creation ioctls and GPU memory DMA mapping. This
creates a hard dependency on specific kernel versions.

### 1.1 Current Kernel Coupling

| Component | Coupling point | Stability |
|-----------|---------------|-----------|
| `snvme` kernel module | Patches stock `nvme` driver internals (struct layout, function signatures) | **Breaks every kernel minor release** |
| `libnvm` user library | Calls snvme ioctls (`NVM_CREATE_QUEUE_GROUP`, `NVM_ADD_USER_QUEUE`, `NVM_MAP_DEVICE_MEMORY`) | Stable (our own ioctls) |
| `nvm_controller_init_b3` | `chrdev_create` + `bind` + `probe` via sysfs | Linux-specific but stable |
| FIEMAP ioctl | `FS_IOC_FIEMAP` for extent extraction | Linux-specific but stable |
| `fallocate` / `O_DIRECT` / `syncfs` | Standard Linux syscalls | Stable |
| `nvidia_p2p_get_pages` | NVIDIA driver API for GPU page pinning | NVIDIA-specific |

### 1.2 Current Version Strategy

The codebase maintains **two directory copies** of the snvme module:

```
backends/local/kernel_modules/
├── snvme-5.15.0-public/          # Ubuntu 22.04 / generic 5.15
├── snvme-5.4.241-1-tlinux4-0017/ # TencentOS 5.4
├── PORTING.md
└── test/
```

There is no build-time version selection, no DKMS integration, and no
automated patch generation. Adding a new kernel version requires
manually porting the patches.

### 1.3 Goals

1. **Isolate kernel-facing logic** so future Linux version support can
   evolve without rewriting upper-layer APIs.
2. **Automate patch maintenance** across kernel versions (DKMS or
   patch-series approach).
3. **Define failure handling** when the module is missing, kernel ABI
   is incompatible, or service bootstrap fails.
4. **Plan an exit strategy** for kernel module dependency (alternative
   approaches that don't require patching the stock driver).

## 2. Kernel API Drift Analysis

The `snvme` module modifies the stock `nvme` driver to add:

1. **User-space queue creation**: `NVM_CREATE_QUEUE_GROUP` +
   `NVM_ADD_USER_QUEUE` ioctls that let user-space create NVMe SQ/CQ
   pairs outside the kernel block layer.
2. **GPU memory DMA mapping**: `NVM_MAP_DEVICE_MEMORY` ioctl that pins
   GPU pages (via `nvidia_p2p_get_pages`) and maps them for NVMe
   controller DMA.
3. **Kernel IOQ cap**: `NVM_SET_KERNEL_IOQ_CAP` to reserve queue pairs
   for the kernel blk-mq path vs. the user QID pool.

These modifications touch internal NVMe driver structures that change
between kernel versions:

| Kernel area | What changes | Frequency |
|-------------|-------------|-----------|
| `struct nvme_dev` | Field additions/reordering | Every major |
| `struct nvme_queue` | Field additions | Every major |
| `nvme_pci_alloc_dev` | Signature changes | Rare |
| `nvme_dev_add` | Refactoring | Occasional |
| `blk_mq` integration | Block layer API evolution | Every major |
| `pci_driver` registration | Stable | Rare |

## 3. Proposed Strategy

### 3.1 Short-Term: Patch-Series + DKMS

Replace the two directory copies with a **single source tree + patch
series** approach:

```
backends/local/kernel_modules/
├── snvme/                    # Single source tree (against latest LTS)
├── patches/                  # Kernel-version-specific patches
│   ├── snvme-base.patch      # Common modifications (applies to all)
│   ├── snvme-5.15.patch      # 5.15-specific adjustments
│   ├── snvme-5.10.patch      # 5.10-specific adjustments
│   ├── snvme-6.1.patch       # 6.1-specific adjustments
│   └── snvme-6.6.patch       # 6.6-specific adjustments
├── dkms.conf                 # DKMS configuration
├── Makefile                  # Build with kernel-version dispatch
└── PORTING.md
```

**Build process:**
```bash
# DKMS auto-build on kernel install:
dkms add backends/local/kernel_modules/snvme
dkms build snvme/<version> -k $(uname -r)
dkms install snvme/<version> -k $(uname -r)

# Or manual:
make KDIR=/lib/modules/$(uname -r)/build KERNEL_VERSION=$(uname -r)
```

**Patch application:**
1. `snvme-base.patch` applies on top of stock `nvme` driver source.
2. If the kernel version has drift, `snvme-<version>.patch` applies on
   top of `snvme-base.patch` to fix struct offsets / API changes.
3. The build system detects the kernel version and selects the right
   patch set.

**Kernel version support matrix:**

| Kernel | Status | Patch file |
|--------|--------|------------|
| 5.4 (TencentOS) | Supported (v0.1) | `snvme-5.4.patch` |
| 5.15 (Ubuntu 22.04) | Supported (v0.1) | `snvme-5.15.patch` |
| 5.10 / 6.1 / 6.6+ | Not supported (v0.1) | Future |

### 3.2 Mid-Term: Kernel Adaptation Layer

Introduce a **kernel adaptation layer** inside the snvme module that
isolates version-sensitive code:

```c
// snvme/adapt/nvme_compat.h

#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 1, 0)
#  define NVME_DEV_QID(dev)        ((dev)->qid)
#  define NVME_QUEUE_SQ(q)         ((q)->sq)
static inline struct nvme_queue *nvme_get_q(struct nvme_dev *dev, int qid) {
    return &dev->queues[qid];
}
#elif LINUX_VERSION_CODE >= KERNEL_VERSION(5, 15, 0)
#  define NVME_DEV_QID(dev)        ((dev)->ctrl.queueid)
#  define NVME_QUEUE_SQ(q)         ((q)->sq_cmds)
static inline struct nvme_queue *nvme_get_q(struct nvme_dev *dev, int qid) {
    return dev->queues[qid];
}
#else
#  define NVME_DEV_QID(dev)        ((dev)->qid)
#  define NVME_QUEUE_SQ(q)         ((q)->sq_cmds)
static inline struct nvme_queue *nvme_get_q(struct nvme_dev *dev, int qid) {
    return &dev->queues[qid];
}
#endif
```

All snvme code uses the compat macros instead of direct field access.
Adding a new kernel version = adding a new `#elif` branch in the compat
header, not rewriting the module.

### 3.3 Backend-Level Approaches

Tutti's design philosophy is **"either a backend is supported, or it
isn't."** Each storage access approach is a distinct backend with its
own implementation. The `snvme` kernel module is THE approach for
`backends/local_nvme/`; other storage access patterns are separate
backends:

| Backend | Kernel module? | GPU-direct? | Status |
|---------|---------------|-------------|--------|
| `local_nvme` (snvme) | Yes (snvme.ko) | Yes (GPU doorbell + GPUDirect DMA) | v0.1 supported |
| `gds` (NVIDIA cuFile) | No (stock kernel) | Yes (via cuFile API) | Planned (Phase 7) |
| `local_rdma` (NVMe-oF / RDMA) | No (stock kernel) | Yes (GPUDirect RDMA) | Community issue |

**Key clarifications:**

1. **snvme is the kernel module for `local_nvme`.** It enables GPU-direct
   NVMe access for local storage. It already provides SPDK-like userspace
   queue capability (CPU-side user-space SQ/CQ pairs via
   `NVM_CREATE_QUEUE_GROUP` + `NVM_ADD_USER_QUEUE`) while coexisting with
   the kernel block layer (via `NVM_SET_KERNEL_IOQ_CAP`).

2. **GDS is a separate backend.** NVIDIA's cuFile API provides GPU-direct
   storage access without a custom kernel module, but only on NVIDIA
   hardware. It is a distinct backend (`backends/gds/`) that implements
   the same `IBackendProvider` SPI.

3. **RDMA is a separate backend for remote/disaggregated storage.** It
   uses stock kernel NVMe-oF or user-space RDMA verbs. GPU submits
   NVMe-oF requests via RDMA QPs mapped to GPU memory. No kernel module
   needed. See [storage-extensibility.md](storage-extensibility.md).

4. **dma-buf is a cross-cutting concern, not a backend.** For
   multi-vendor GPU support, the `snvme` kernel module's
   `NVM_MAP_DEVICE_MEMORY` ioctl will be extended per vendor (see
   [gpu-abstraction.md](gpu-abstraction.md) §5.3). The Linux dma-buf
   framework may be used as the vendor-neutral sharing mechanism inside
   the kernel module, but it is not a separate backend.

## 4. User-Space Isolation

The user-space library (`libnvm`) and all upper layers must NOT depend
on kernel version. The kernel-facing logic is isolated in:

1. **`snvme` module** (kernel space): version-specific, adapted via
   compat layer.
2. **`libnvm`** (user space): talks to snvme via stable ioctls. The
   ioctl ABI is our own and does not change with kernel versions.
3. **`device_manager`** (runtime): calls `libnvm` functions, never
   touches kernel APIs directly.

```
Runtime (device_manager, nvme_storage, ...)
    │  stable API
    ▼
libnvm (user space)
    │  stable ioctl ABI (our own)
    ▼
snvme.ko (kernel space)
    │  version-adapted via compat layer
    ▼
stock nvme driver (kernel space)
```

The only version-sensitive code is inside `snvme.ko`. If the kernel
upgrades and snvme doesn't compile, the runtime API is unaffected —
only the module needs a patch.

## 5. Deployment and Failure Handling

### 5.1 Prerequisites Check

Before data-path use, the runtime MUST verify:

```cpp
// device_manager/src/prerequisites.cpp
bool check_runtime_prerequisites() {
    // 1. snvme module loaded?
    if (!file_exists("/dev/snvm_control")) return error("snvme not loaded");

    // 2. Kernel version supported?
    if (!is_supported_kernel(uname())) return error("kernel not supported");

    // 3. NVMe device bound to snvme (not stock nvme)?
    if (!is_bound_to_snvme(pci_addr)) return error("NVMe bound to stock nvme");

    // 4. GPU driver loaded (for GPUDirect)?
    if (!file_exists("/proc/driver/nvidia")) return error("NVIDIA driver not loaded");

    // 5. IOMMU = passthrough?
    if (!is_iommu_pt()) return error("IOMMU must be in passthrough mode");

    return true;
}
```

### 5.2 Failure Modes

| Failure | Detection | Behavior |
|---------|-----------|----------|
| snvme not loaded | `/dev/snvm_control` missing | Runtime refuses to bootstrap; prints install instructions |
| Kernel ABI incompatible | snvme ioctl returns `-ENOTTY` | Runtime logs error, suggests DKMS rebuild |
| NVMe bound to stock nvme | sysfs driver check | Runtime prints unbind instructions |
| GPU driver not loaded | `/proc/driver/nvidia` missing | Runtime refuses GPU_SUBMIT; CPU_SUBMIT via a non-GPUDirect backend may be available |
| IOMMU not passthrough | `/sys/kernel/iommu_groups` check | Runtime logs error; GPUDirect requires IOMMU=pt |

### 5.3 Deployment Models

1. **System startup installation** (preferred):
   - `dkms install snvme` during system setup
   - `systemd` service to bind NVMe devices to snvme at boot
   - NVMeService daemon started after snvme load

2. **Ad hoc runtime build/load** (development only):
   - `make && insmod snvme.ko` in a script
   - Not recommended for production

3. **Pre-built package** (distribution):
   - Ship pre-built `snvme.ko` for supported kernel versions
   - Package manager handles DKMS rebuild on kernel upgrade
