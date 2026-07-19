# Tutti System Architecture

> **Status**: v0.1 baseline. This document describes the architecture as
> implemented in the current codebase, with explicit notes where the
> implementation diverges from the `Roadmap.md` target.

## 1. Overview

Tutti (Italian for "all instruments together") is a **CPU/GPU companion
storage software stack**: a unified storage runtime in which the CPU and
GPU paths cooperate on top of a shared memory subsystem and a pluggable
backend SPI.

The core idea: **the CPU launches IO kernels, the GPU executes them.**
The CPU prepares batch descriptors, stages them into GPU memory, and
launches a GPU kernel that rings NVMe doorbells directly — bypassing the
host CPU on the data path. This delivers GPU-direct storage access
without requiring the CPU to mediate every IO.

### 1.1 Design Goals

| Goal | How |
|------|-----|
| GPU-direct NVMe access | Modified kernel module (`snvme`) + `libnvm` user library; GPU kernels write SQ entries and tap doorbells directly |
| Batch IO throughput | One GPU kernel launch handles thousands of NVMe IOs; per-IO shard selection happens on-GPU |
| Multi-device striping | A `GpuFile` spans up to 4 NVMe devices (shards); data is interleaved across them |
| Pluggable backends | `IBackendProvider` SPI isolates the transport (NVMe today, RDMA/GDS future) from upper layers |
| Two bootstrap modes | Direct (process owns NVMe) or service-client (daemon owns NVMe, process attaches) |

### 1.2 Current Scope (v0.1)

- **One backend**: `local_nvme` (modified snvme kernel module + libnvm + NVMeService daemon)
- **One accelerator vendor**: NVIDIA CUDA
- **One filesystem path**: ext4 + FIEMAP extent extraction
- **Two submission modes**: `BATCH_GPU_STREAM` (GPU submits) and `BATCH_CPU_SYNC` (CPU submits, blocks)
- **No cooperative submit** (interface defined, not implemented)

## 2. Layered Architecture

The codebase is organized into the following layers. Each layer depends
only on the layer(s) below it and on shared type headers.

```
┌─────────────────────────────────────────────────────────┐
│                   Application / Adapter                   │
│              (examples/, adapters/)                       │
├─────────────────────────────────────────────────────────┤
│                     Coordinator                           │
│  (coordinator/) — owns one instance of every layer,       │
│  sequences bring-up/tear-down, provides unified API       │
├──────────────┬───────────────┬──────────────────────────┤
│  io_engine   │   memory      │   device_manager          │
│ (io_engine/) │  (memory/)    │ (device_manager/)         │
│              │               │                           │
│ IIoEngine    │ IMemorySub-   │ IDeviceRegistry           │
│ IBackend-    │ system        │ ILeaseManager             │
│  Provider    │ MemoryRegion  │ NvmeQueueGroup            │
│              │               │                           │
├──────────────┴───────────────┴──────────────────────────┤
│              block_storage (block_storage/)               │
│  IBlockStorage — GpuFile = N NvmeFile shards              │
├─────────────────────────────────────────────────────────┤
│              nvme_storage (nvme_storage/)                 │
│  INvmeStorage — NvmeFile, FIEMAP extents, GPU handle,     │
│  device-side submit_read_one / submit_write_one           │
├─────────────────────────────────────────────────────────┤
│              backends/local (backends/local/)             │
│  kernel_modules/  — snvme (modified NVMe kernel module)   │
│  nvme/libnvm/     — user-space NVMe queue library         │
│  NVMeService/     — daemon for persistent device mount    │
└─────────────────────────────────────────────────────────┘
```

### 2.1 Layer Responsibilities

#### Coordinator (`coordinator/`)

The top-level orchestrator. Owns one instance of every layer below and
sequences their bring-up / tear-down so application code never assembles
the chain by hand.

**Bring-up order:**
```
IDeviceRegistry          (device_manager)
  → HostFsBackedNvmeStorage   (nvme_storage)
    → HostFsBackedBlockStorage  (block_storage)
      → HostDeviceMemorySubsystem (memory)
        → LocalNvmeIoEngine        (io_engine)
```

**Key types**: `Coordinator`, `CoordinatorConfig`, `Device`, `CapabilitySet`, `Lease`

**API surface**: file lifecycle (open/delete/batch), handle cache
(id→handle resolution), memory (allocate/register/free), IO submission
(submit_batch), durability (sync_file).

#### IO Engine (`io_engine/`)

Defines the IO submission interfaces and batch metadata. Two levels:

- **`IIoEngine`** — host-side entry point. `submit_batch()` takes a
  vector of `NvmeBatchInputTensor` (tensor region + file handle +
  offset), builds the batch on host, `cudaMemcpyAsync` to GPU, launches
  the transfer kernel, and synchronizes.

- **`IBackendProvider`** — backend SPI. Defines four submission modes
  (`BATCH_GPU_STREAM`, `BATCH_CPU_SYNC`, `BATCH_CPU_ASYNC`, `COOP`),
  descriptor preparation, queue acquisition, and lifecycle hooks. Each
  backend implements this interface.

**Key types**: `IIoEngine`, `IBackendProvider`, `IQueue`, `IQueueProvider`,
`BufferDescriptor` (tagged union: NVMe/RDMA), `IORequest`, `IORequestBatch`,
`IOCompletion`, `IOFuture`, `CoopIOChannel`, `IOSubmitMode`, `BackendType`

**GPU dispatch model**: GPU kernels are NOT polymorphic — there is no
vtable on GPU. The host picks the backend and calls
`launch_batch_gpu_stream()`; the backend's host-side method launches its
own `__global__` kernel directly. GPU code reads queue ring pointers
(`QueueDesc::sq.dev_ptr` / `cq.dev_ptr`) and calls backend-private
device-side helpers.

#### Memory (`memory/`)

The single source of truth for "the runtime knows about this piece of
memory" and for translating tensors into NVMe-readable address
descriptors (PRP today, SGL when supported).

**Responsibilities:**
- Allocate host / pinned-host / device / managed memory
- Register caller-allocated buffers (including external: CUDA IPC, shm, app-managed)
- DMA-map registered buffers to NVMe controllers (via libnvm `nvm_dma_map_data_device`)
- Pre-compute IO-slice tables: split tensors into `granularity`-byte slices, each fanned into NVMe-sized sub-IOs with pre-built PRP entries
- Two-tier PRP-page cache (GPU-resident L1 + host-pinned L2) to avoid per-tensor allocation waste

**Key types**: `IMemorySubsystem`, `HostDeviceMemorySubsystem`,
`MemoryRegion`, `MemoryKind`, `TensorRegistrationSpec`, `IoSliceView`,
`AddressDescriptor`, `DescriptorFormat`, `PrpPageCache`

**Cluster-wide invariant**: The descriptor format (PRP vs SGL) is frozen
at bootstrap. Under IOMMU=pt + bare metal, GPU physical pages have
controller-agnostic PCI bus addresses, so one PRP buffer + one descriptor
blob serve every cluster-bound NVMe controller.

#### Device Manager (`device_manager/`)

Cross-process device fleet management. Discovers, registers, and leases
storage devices.

**Responsibilities:**
- Device discovery and registry (`IDeviceRegistry`)
- Two implementations: `LocalNvmeDirectRegistry` (process owns NVMe) and `NvmeServiceBackedRegistry` (daemon owns, process attaches)
- GPU-resident NVMe queue pool (`NvmeQueueGroup`): creates N user-queue pairs, stages `QueuePair[]` on GPU
- Lease lifecycle (`ILeaseManager`): heartbeat-managed resource tokens for multi-process sharing

**Key types**: `IDeviceRegistry`, `LocalNvmeDevice`, `NvmeQueueGroup`,
`ILeaseManager`, `LocalNvmeDirectRegistry`, `NvmeServiceBackedRegistry`

#### Block Storage (`block_storage/`)

Owns the "GpuFile" abstraction: a named logical container whose data
plane is split across N NvmeFile shards (one per device, up to
`kGpuFileMaxShards=4`).

**Responsibilities:**
- GpuFile directory (create/open/delete, batch variants)
- Metadata persistence (`gpu_file_log.bin` per device)
- Handle acquisition: brings all N shard NvmeFiles to GPU together, produces `GpuFileHandle` with `d_shards_dev` pointer table
- Stripe resolution: `gpu_file_resolve()` maps a global byte offset to (shard index, shard-local offset)

**Key types**: `IBlockStorage`, `GpuFile`, `GpuFileSpec`, `GpuFileHandle`

**Data layout**: Data is interleaved across shards in `tensor_size`-byte
units. For KV-cache: `num_shards=2` (K, V) or `4` (K_lo, K_hi, V_lo,
V_hi); `tensor_size` = per-layer tensor size.

#### NVMe Storage (`nvme_storage/`)

The lowest storage abstraction above the raw backend. Owns named LBA
ranges on top of NVMe namespaces.

**Responsibilities:**
- Mount host filesystem (ext4) on the snvme block device
- Create files via `fallocate`, extract physical LBA extents via FIEMAP ioctl
- Maintain persistent file directory (`PersistentFileLog`)
- Provide host-side blocking IO (O_DIRECT pread/pwrite) for bootstrap/metadata/tests
- Provide GPU-side device handle (`NvmeFileDeviceHandle`): a small POD in GPU memory carrying file extents + queue pool pointer
- Device-side submit kernels: `submit_read_one` / `submit_write_one` (`__device__` functions that ring NVMe doorbells)
- Two-tier handle cache (GPU-resident L1 + CPU-pinned L2) for millions of files

**Key types**: `INvmeStorage`, `NvmeFile`, `NvmeFileDeviceHandle`,
`LbaExtent`, `PersistentFileLog`

**NvmeFileDeviceHandle design**: Inline 8 extents (common case: 1-3
extents from unfragmented fallocate); overflow buffer for pathological
fragmentation. Mirrors NVMe's PRP1/PRP2 + PRP-list pattern. ~200 bytes
per file vs ~2 KiB naive — critical for LMCache-scale (millions of files).

#### Backend: local (`backends/local/`)

The reference (and currently only) backend implementation.

- **`kernel_modules/`** — Modified Linux NVMe kernel module (`snvme`).
  Patches the stock `nvme` driver to expose user-space queue creation
  ioctls (`NVM_CREATE_QUEUE_GROUP`, `NVM_ADD_USER_QUEUE`) and GPU memory
  DMA mapping (`NVM_MAP_DEVICE_MEMORY`). Two kernel versions supported:
  5.15.0 and 5.4.241.

- **`nvme/libnvm/`** — User-space C++ library wrapping the snvme
  ioctls. Provides `nvm_ctrl_t` (controller handle), `nvm_dma_t` (DMA
  mapping), `QueuePair` (SQ/CQ ring management), and the B3 bootstrap
  path (`nvm_controller_init_b3`: chrdev_create + capability + bind +
  probe).

- **`NVMeService/`** — Daemon for persistent NVMe device mounting. Owns
  chrdev/bind lifecycle, leases queue ranges to client processes via
  gRPC, and reaps dead-client queues via PID-starttime tracking. Clients
  attach via `nvm_ctrl_attach_client` and build their own user queues.

## 3. Core Data Flow

### 3.1 GPU-Submit Read Path (Primary High-Throughput Path)

```
Application                Coordinator              io_engine            GPU Kernel
    │                           │                       │                    │
    │ register_tensor(buf)      │                       │                    │
    │──────────────────────────>│                       │                    │
    │                           │ memory.register_tensor │                    │
    │                           │  DMA-map to NVMe       │                    │
    │                           │  build IO-slice table  │                    │
    │                           │  admit PRP pages to L2 │                    │
    │                           │                       │                    │
    │ open_gpu_file(spec)       │                       │                    │
    │──────────────────────────>│                       │                    │
    │   GpuFile*                │ block_storage          │                    │
    │<──────────────────────────│  .open (creates shards)│                    │
    │                           │                       │                    │
    │ handle_for(file_id, stream)│                      │                    │
    │──────────────────────────>│                       │                    │
    │   GpuFileHandle*          │ block_storage          │                    │
    │<──────────────────────────│  .acquire_device_handle│                    │
    │                           │  nvme_storage          │                    │
    │                           │   .acquire (FIEMAP→GPU)│                    │
    │                           │                       │                    │
    │ submit_batch(inputs, READ, stream)                │                    │
    │──────────────────────────>│                       │                    │
    │                           │ io_engine.submit_batch │                    │
    │                           │  build NvmeBatchEntry[]│                    │
    │                           │  ensure_prp_pages (L2→L1)                   │
    │                           │  cudaMemcpyAsync H2D   │                    │
    │                           │  launch kernel ──────────────────────────>  │
    │                           │                       │  per-thread:       │
    │                           │                       │   gpu_file_resolve │
    │                           │                       │   resolve_lba      │
    │                           │                       │   acquire_queue    │
    │                           │                       │   issue NVMe cmd   │
    │                           │                       │   poll completion  │
    │                           │  cudaStreamSynchronize │                    │
    │<──────────────────────────│                       │                    │
```

**Key steps:**
1. **Register tensor**: Memory layer DMA-maps the buffer to NVMe, pre-computes IO-slice table (per-slice PRP entries), admits PRP-list pages to L2 cache.
2. **Open file**: Block storage creates NvmeFile shards (fallocate + FIEMAP), records in persistent log.
3. **Acquire handle**: NVMe storage builds `NvmeFileDeviceHandle` (extents + queue pool ptr), cudaMalloc + cudaMemcpy to GPU. Cached in two-tier cache.
4. **Submit batch**: IO engine flattens (tensor × sub-slice) → `NvmeBatchEntry[]`, ensures PRP pages are L1-resident, H2D copies entries, launches `nvme_batch_xfer_kernel`.
5. **GPU kernel**: Each thread resolves its IO to (shard, LBA, PRP), acquires a queue, issues NVMe command, busy-polls completion.

### 3.2 CPU-Submit Path

CPU prepares descriptors and submits directly via `IQueue::enqueue_cpu`,
then polls `IQueue::poll_cq_cpu`. No GPU kernel involved. Used for
bootstrap, metadata, and testing.

## 4. Key Abstractions

### 4.1 Device

```cpp
struct Device {
    int32_t      device_id;        // dense index inside the Runtime
    BackendType  backend_type;     // LOCAL_NVME | RDMA | GDS
    std::string  pci_addr;         // PCI BDF / IB GUID / ...
    std::string  display_name;
    CapabilitySet capabilities;
    void* backend_private;         // backend-specific (e.g. LocalNvmeDevice*)
};
```

A `Device` is the runtime-visible identity of one storage resource.
Upper layers refer to devices via `device_id` and `Device*` handles
obtained from the registry. The `backend_private` field lets backends
stash a private handle (e.g. `LocalNvmeDevice*`) to navigate back to
concrete state.

> **Note**: Resolved — `runtime/` has been deleted (see §9 item 1).
> `Device` now has a single canonical definition here in
> `coordinator/include/`; there is no longer a duplicate `device.h`.

### 4.2 MemoryRegion

```cpp
struct MemoryRegion {
    uint64_t   region_id;
    MemoryKind kind;              // HOST | PINNED_HOST | DEVICE | MANAGED | EXTERNAL
    int        cuda_device;
    void*    host_ptr;            // CPU-visible base
    void*    device_ptr;          // GPU-visible base
    uint64_t size;
    ExternalMemorySpec external;  // for EXTERNAL kind
    RegistrationMetadata registration;  // DMA ioaddrs, RDMA keys
};
```

The runtime-internal handle for registered memory. Every layer that
needs to refer to memory carries `const MemoryRegion*` instead of raw
pointers + DMA addresses. Ownership stays with the application;
`unregister()` drops metadata but leaves the buffer untouched.

### 4.3 GpuFile / NvmeFile

- **`NvmeFile`**: One file on one NVMe device. Metadata-only (no held
  fd). Carries FIEMAP-extracted `LbaExtent[]` (physical LBAs).

- **`GpuFile`**: A logical container spanning N `NvmeFile` shards. Data
  is interleaved in `tensor_size`-byte units across shards.

- **`GpuFileHandle`**: Acquired view — every shard has a
  `NvmeFileDeviceHandle*` in GPU memory, plus a GPU-resident pointer
  table `d_shards_dev`.

### 4.4 BufferDescriptor (Tagged Union)

```cpp
struct BufferDescriptor {
    BackendType backend_type;     // discriminator
    union {
        NVMeBufferDesc nvme;      // PRP/SGL addressing
        RDMABufferDesc rdma;      // remote/local addr + rkey/lkey
        uint8_t raw[48];          // 48-byte budget for future backends
    };
};
```

Backend-neutral wrapper for per-slice IO metadata. Filled by
`IBackendProvider::prepare_descriptors()` during tensor registration.
Device-side kernels read the union member matching their backend.

### 4.5 CapabilitySet

Bitmask advertising what a Device supports:
- **Submission modes** (bits 0-15): `CAP_SUBMIT_BATCH_GPU_STREAM`, `CAP_SUBMIT_BATCH_CPU_SYNC`, `CAP_SUBMIT_COOP_*`, etc.
- **Memory sources** (bits 16-23): `CAP_MEM_HOST_REGISTER`, `CAP_MEM_DEVICE_REGISTER`, `CAP_MEM_EXTERNAL_*`, etc.
- **Transport/topology** (bits 24-31): `CAP_TOPO_GPUDIRECT_DMA`, `CAP_TOPO_GPUDIRECT_RDMA`, `CAP_TOPO_NUMA_AWARE`

## 5. Bootstrap Modes

### 5.1 IN_PROCESS (Direct)

```
Application Process
  ├── LocalNvmeDirectRegistry::Open()
  │     nvm_controller_init_b3(pci_addr)  // chrdev_create + bind + probe
  │     NvmeQueueGroup(ctrl, ...)         // create user queues, stage on GPU
  │     → Device { backend_private = LocalNvmeDevice* }
  ├── HostFsBackedNvmeStorage::bootstrap(devices)
  │     mount ext4 on /dev/snvmesdX
  │     load PersistentFileLog
  ├── HostFsBackedBlockStorage::bootstrap(storage, devices)
  │     load gpu_file_log.bin
  ├── HostDeviceMemorySubsystem::bind_devices(devices)
  │     cache cluster caps (page_size, min MDTS)
  └── LocalNvmeIoEngine(mem, max_entries)
```

### 5.2 SERVICE_CLIENT (Daemon-Attached)

```
NVMeService Daemon (started beforehand)
  ├── owns chrdev_create + bind for every NVMe
  ├── ListDevices() → fleet
  └── Connect(client) → lease queue range, heartbeat-managed

Application Process
  ├── NvmeServiceBackedRegistry::Open()
  │     gRPC connect to daemon
  │     ListDevices → pick daemon_device_ids
  │     Connect per device → session + lease
  │     nvm_ctrl_attach_client(fd)       // attach to daemon-owned ctrl
  │     NvmeQueueGroup(ctrl, ...)        // client builds own user queues
  │     → Device { backend_private = LocalNvmeDevice* }
  └── (rest identical to IN_PROCESS)
```

Key difference: the daemon owns chrdev/bind; the client only attaches.
No GPU memory or queue handles cross the process boundary. The daemon
brokers the lease; the client runs `nvm_create_group` +
`nvm_add_user_queue` against its own attach_client fd.

## 6. GPU-Side IO Submission

The GPU kernel is the heart of the data path. It is NOT polymorphic —
there is no vtable on GPU.

### 6.1 Kernel Flow (`nvme_batch_xfer_kernel`)

```
For each thread tid (one per NvmeBatchEntry):
  1. Load entry = d_entries[tid]
  2. gpu_file_resolve(tensor_size, num_shards, file_offset, &shard_idx, &shard_off)
  3. shard_handle = entry.shards[shard_idx]   // NvmeFileDeviceHandle*
  4. For each sub-slice i in [0, entry.num_sub_slices):
       prp = entry.prp_entry[entry.prp_idx + i]
       resolve_lba(shard_handle, shard_off + i*io_size, io_size, &lba, &n_blocks)
       qidx = QueueAcquireHelper::acquire_queue(num_d_qps)
       qp = &shard_handle->d_qps[qidx]
       QueueAcquireHelper::issue_nvme_cmd(qp, prp.prp1, prp.prp2, n_blocks, lba, opcode, &cid)
       QueueAcquireHelper::poll(qp, cid)   // busy-poll completion
```

### 6.2 Queue Acquisition

`QueueAcquireHelper` (device-side) provides:
- `acquire_queue(num_d_qps)`: atomically claims a queue index (round-robin with atomic counter)
- `issue_nvme_cmd(qp, prp1, prp2, n_blocks, lba, opcode, &cid)`: writes SQE, advances SQ tail, rings doorbell
- `poll(qp, cid)`: busy-polls CQ for matching command ID

### 6.3 LBA Resolution

`resolve_lba()` walks `NvmeFileDeviceHandle::extents[]` (inline 8 +
overflow) to map (logical_offset, nbytes) → (starting_lba, n_blocks).
Rejects requests spanning multiple extents (R7 pre-splits per-extent
before calling submit).

## 7. Memory Model

### 7.1 Memory Kinds

| Kind | Source | GPU-visible | NVMe-DMA-mappable |
|------|--------|-------------|-------------------|
| `HOST` | malloc/new | No | Via nvm_dma_map_data_host |
| `PINNED_HOST` | cudaMallocHost | Yes (cudaHostGetDevicePointer) | Yes |
| `DEVICE` | cudaMalloc | Yes | Via nvm_dma_map_data_device |
| `MANAGED` | cudaMallocManaged | Yes (unified VA) | Yes |
| `EXTERNAL` | App-supplied (IPC/shm/slab) | Per ExternalMemorySpec | Per source |

### 7.2 IO-Slice Table

At `register_tensor()` time, the memory layer pre-computes:
1. Split tensor into `granularity`-byte slices
2. Each slice fans into `ceil(granularity / effective_io)` sub-IOs, where `effective_io = min(granularity, min_mdts)`
3. Each sub-IO gets a pre-built `AddressDescriptor` (prp1, prp2, data_length)
4. Upload the entire descriptor blob to GPU memory
5. Build PRP-list pages (when sub-IO needs > 2 pages) — cached in two-tier PRP page cache

The IO engine looks up slices in O(log N) via `lookup_io_slice()` instead
of recomputing PRPs on the hot path.

### 7.3 Two-Tier Handle Cache

For LMCache-scale workloads (millions of files), holding every file's
GPU handle in GPU memory is infeasible. The cache has two tiers:

- **L1 (GPU-resident, DMA-mapped)**: Small working set. Sized by
  `handle_l1_gpu_budget_bytes` (default 512 MiB). Eviction = downgrade
  to L2 (content preserved, not deleted).

- **L2 (CPU-pinned)**: Large. Sized by `handle_l2_host_budget_bytes`
  (default 2 GiB ≈ 11M files). Every touched file's built template
  lives here. L1→L2 promotion is a plain memcpy, never re-walks FIEMAP.

## 8. Current Directory Structure

```
Tutti/
├── coordinator/          # Top-level orchestrator + runtime nouns (Device, Lease, etc.)
│   └── include/          #   coordinator.h, coordinator_config.h, device.h, capability_set.h, lease.h
├── io_engine/            # IO submission SPI + batch metadata
│   └── include/
│       ├── local_nvme/   #   NVMe-specific: host_batch_builder, launch_batch, nvme_batch, io_engine impl
│       ├── io_engine.h, backend_provider.h, buffer_descriptor.h, io_request.h
│       ├── queue.h, queue_provider.h, io_future.h, coop_channel.h
│       └── io_submit_mode.h, backend_type.h
├── memory/               # Memory subsystem: registration, DMA mapping, PRP/SGL, IO-slice tables
│   └── include/          #   memory_subsystem.h, memory_region.h, host_device_memory_subsystem.h, ...
├── device_manager/       # Device discovery, registry, lease, GPU queue pool
│   └── include/          #   device_registry.h, local_nvme_device.h, nvme_queue_group.h, lease_manager.h, ...
├── block_storage/        # GpuFile abstraction (multi-shard), directory, handle cache
│   └── include/          #   block_storage.h, gpu_file_resolve.h, ...
├── nvme_storage/         # NvmeFile, FIEMAP extents, GPU device handle, device-side submit kernels
│   └── include/          #   nvme_storage.h, nvme_file.h, nvme_file_device_handle.h, nvme_storage_device.cuh, ...
├── backends/
│   └── local/            # The local_nvme backend
│       ├── kernel_modules/  # snvme (modified NVMe kernel module, 2 kernel versions)
│       ├── nvme/libnvm/     # User-space NVMe queue library
│       └── NVMeService/     # Daemon for persistent device mounting
├── runtime/              # (Largely unused — only device.h + capability_set.h, duplicated from coordinator/)
├── adapters/             # Framework integration adapters (LMCache, Mooncake, ...)
├── examples/             # Smoke tests and examples
├── scripts/              # Setup and bind scripts
├── doc/                  # Documentation
├── CMakeLists.txt
├── sys_config.yaml
├── Roadmap.md
└── README.md
```

## 9. Known Architecture Issues (Current vs Target)

The current codebase does not yet match the `Roadmap.md` v0.1 target
architecture. Main gaps (with resolution status):

1. **~~`runtime/` was essentially empty~~** — RESOLVED. The `runtime/`
   directory has been deleted. Runtime nouns (`Device`, `CapabilitySet`,
   `Lease`) stay in `coordinator/include/` as the canonical home.

2. **No `filesystems/` layer.** FIEMAP extent extraction is baked into
   `nvme_storage`, making it impossible to pair the same NVMe backend
   with a different namespace resolver (e.g. raw block range, custom
   on-device layout, distributed FS client).
   → Resolution: [Phase 2](../refactor/restructuring-plan.md#phase-2)

3. **`io_engine`'s `IIoEngine` is NVMe-specific.** It takes
   `NvmeBatchInputTensor` directly, not the generic
   `BufferDescriptorBatch` + `IORequestBatch` defined in the SPI. The
   generic SPI (`IBackendProvider`) exists but is not yet wired into
   the runtime — `LocalNvmeIoEngine` bypasses it.
   → Resolution: [Phase 4](../refactor/restructuring-plan.md#phase-4)

4. **CUDA is hardcoded throughout.** `cuda_runtime.h` appears in
   coordinator, io_engine, memory, and device_manager public headers.
   `cudaStream_t`, `cudaMalloc`, `cudaFree` are used directly in
   interfaces. There is no accelerator abstraction layer.
   → Resolution: [Phase 1](../refactor/restructuring-plan.md#phase-1)

5. **libnvm leaks into non-backend headers.** `nvm_types.h` /
   `nvm_ctrl_t` / `nvm_dma_t` appear in `memory/` and
   `device_manager/` headers, not just in `backends/local/`.
   → Resolution: [Phase 3](../refactor/restructuring-plan.md#phase-3)

6. **Kernel module versioning is ad-hoc.** Two directory copies
   (`snvme-5.15.0-public`, `snvme-5.4.241-1-tlinux4-0017`) with no
   build-time version selection or DKMS strategy.
   → Resolution: [Phase 6](../refactor/restructuring-plan.md#phase-6)

7. **File is the only storage abstraction.** No `StorageTarget` type;
   can't do raw NVMe or RDMA without files.
   → Resolution: [Phase 5](../refactor/restructuring-plan.md#phase-5)

These issues and their resolution are addressed in the
[Refactoring Plan](../refactor/restructuring-plan.md) and related design
documents ([gpu-abstraction](../design/gpu-abstraction.md),
[kernel-portability](../design/kernel-portability.md),
[storage-extensibility](../design/storage-extensibility.md)).
