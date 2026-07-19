# Storage Extensibility Design

> **Status**: Design proposal. Not yet implemented. This document
> addresses three extensibility requirements:
> 1. RDMA backend (GPU constructs and submits NVMe-over-RDMA requests
>    via QPs mapped to GPU memory)
> 2. Raw NVMe access (direct NVMe command submission without files)
> 3. File as optional storage type (files are one option, not the only
>    abstraction)

## 1. Problem Statement

The current codebase is built around **files on local NVMe**. Every IO
goes through:
`GpuFile → NvmeFile (shard) → FIEMAP extents → LBA → NVMe command`

This is too narrow for three classes of future users:

1. **RDMA users**: Some deployments want to access remote storage via
   NVMe-oF (NVMe over Fabrics) over RDMA. The GPU should be able to
   construct NVMe-oF RDMA requests directly and submit them via RDMA
   QPs mapped into GPU memory — without the CPU mediating.

2. **Raw NVMe users**: Some users want to submit NVMe commands directly
   against a namespace (LBA range), without any filesystem or file
   abstraction. Files are unnecessary overhead for block-oriented
   workloads (e.g., database pages, raw KV stores).

3. **File-optional users**: Files should be one storage object form,
   not the only storage abstraction. The runtime should treat a file,
   a raw block range, and a remote object as different `StorageTarget`
   shapes, all consumable by the same backend.

## 2. Current File-Centric Architecture

```
Application
  │
  ▼
GpuFile (block_storage)      ← file is mandatory
  │  1 GpuFile = N NvmeFile shards
  ▼
NvmeFile (nvme_storage)      ← file is mandatory
  │  1 NvmeFile = host fs file + FIEMAP extents
  ▼
LBA range on NVMe namespace
  │
  ▼
NVMe command (snvme + libnvm)
```

FIEMAP extent extraction is baked into `nvme_storage`. The file path is
the only way to reach NVMe LBAs. There is no `StorageTarget` abstraction
— the `GpuFileHandle` directly carries `NvmeFileDeviceHandle*` pointers.

## 3. Proposed Storage-Target Architecture

### 3.1 Two Extension Axes

Separate **transport** (how bytes move) from **namespace** (how names
resolve to addresses):

```
                     StorageTarget
                    (meeting point)
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
  filesystems/                      backends/
  (namespace → address)           (transport → bytes)
         │                               │
  ┌──────┼──────┐               ┌───────┼───────┐
  │      │      │               │       │       │
  ext4   tutti  dfs     +   local   local    gds
  fiemap layout client   _nvme   _rdma
  │      │      │               │       │       │
  └──────┴──────┘               └───────┴───────┘
  produce StorageTarget         consume StorageTarget
```

- `filesystems/` resolves names (file paths, object keys, DFS handles)
  into `StorageTarget` values.
- `backends/` consumes `StorageTarget` values and moves bytes.
- They meet only through `StorageTarget`. Adding one transport does not
  require touching any filesystem code, and vice versa.

### 3.2 StorageTarget Type

```cpp
// runtime/include/storage_target.h

enum class StorageTargetKind : uint32_t {
    NVME_FILE     = 0,   // file on local NVMe (FIEMAP extents)
    NVME_RAW      = 1,   // raw LBA range on local NVMe namespace
    RDMA_REMOTE   = 2,   // remote storage via RDMA (NVMe-oF or raw RDMA)
    OBJECT_KEY    = 3,   // object store key (future)
};

struct StorageTarget {
    StorageTargetKind kind;
    BackendType        backend_type;     // LOCAL_NVME | RDMA | GDS
    int32_t            device_id;        // which Device this target lives on

    union {
        // NVME_FILE: file_id + FIEMAP-extracted LBA extents
        struct {
            uint64_t       file_id;
            uint32_t       num_extents;
            const LbaExtent* extents;    // points to cached extent array
            uint32_t       namespace_id;
            uint32_t       block_size;
        } nvme_file;

        // NVME_RAW: direct LBA range, no file
        struct {
            uint32_t namespace_id;
            uint64_t start_lba;
            uint64_t length_blocks;
            uint32_t block_size;
        } nvme_raw;

        // RDMA_REMOTE: remote address + memory key
        struct {
            uint64_t remote_addr;
            uint32_t rkey;
            uint32_t reserved;
        } rdma_remote;
    };
};
```

### 3.3 Filesystem Layer

```cpp
// filesystems/include/filesystem.h

class IFilesystem {
public:
    virtual ~IFilesystem() = default;

    /// Resolve a name to a StorageTarget.
    /// For ext4_fiemap: open(path) → FIEMAP → (file_id, extents)
    /// For tutti_layout: read on-device header → (file_id, extents)
    /// For raw/passthrough: return nvme_raw directly (no FS involved)
    virtual bool resolve(const std::string& name, StorageTarget* out) = 0;

    /// Create a named target (for file-backed filesystems).
    /// For raw/passthrough: no-op (targets are address-based, not named).
    virtual bool create(const std::string& name, uint64_t size,
                        StorageTarget* out) = 0;

    /// Delete a named target.
    virtual bool remove(const std::string& name) = 0;

    /// List all known names.
    virtual std::vector<std::string> list() const = 0;
};
```

**Implementations:**

| Filesystem | How it resolves names | Backend compatibility |
|-----------|----------------------|----------------------|
| `ext4_fiemap/` | `open(path)` + `FS_IOC_FIEMAP` → LBA extents | local_nvme, gds |
| `tutti_layout/` | Custom on-device header → LBA extents | local_nvme |
| `dfs_client/` | Distributed FS client (3FS, JuiceFS, DAOS) → block range | local_rdma, gds |
| `raw_passthrough/` | No resolution; caller provides `(namespace, LBA range)` directly | local_nvme |

**Raw passthrough**: For users who want direct NVMe access without files,
the `raw_passthrough/` filesystem is a no-op — the caller constructs
`StorageTarget { kind = NVME_RAW, nvme_raw = { ... } }` directly and
never calls `IFilesystem::resolve()`. This makes files completely
optional.

### 3.4 Backend Consumes StorageTarget

```cpp
class IBackendProvider {
    // ... existing methods ...

    /// Build a device-resident handle from a StorageTarget.
    /// For local_nvme: cudaMalloc NvmeFileDeviceHandle, fill extents.
    /// For RDMA: register remote memory, build RDMABufferDesc.
    virtual void* acquire_target_handle(const StorageTarget& target,
                                         AccelStream stream) = 0;

    /// Release a target handle.
    virtual void release_target_handle(void* handle, AccelStream stream) = 0;
};
```

The backend converts `StorageTarget` into its own device-resident handle
(e.g., `NvmeFileDeviceHandle*` for local_nvme, `RdmaTargetHandle*` for
RDMA). The IO engine carries these handles in `IORequest::descriptor`.

## 4. RDMA Backend Design

### 4.1 GPU-Direct RDMA Submission

The goal: GPU kernels construct NVMe-oF RDMA requests and submit them
directly via RDMA QPs mapped into GPU memory — without CPU mediation.

```
GPU Memory
┌─────────────────────────────────────┐
│  RDMA QP rings (SQ/CQ)              │  ← mapped to GPU via
│  RDMA Work Requests                 │     ibv_reg_mr + GPUDirect RDMA
│  NVMe-oF command capsules           │
└──────────────────┬──────────────────┘
                   │ GPU writes SQ entry + doorbell
                   ▼
            RDMA NIC (RoCE / CX-7)
                   │ RDMA send to NVMe-oF target
                   ▼
            NVMe-oF Target (local or remote)
                   │ NVMe command to NVMe SSD
                   ▼
            NVMe SSD
```

### 4.2 Components

```
backends/local_rdma/
├── include/
│   ├── rdma_backend.h           # IBackendProvider impl
│   ├── rdma_queue_group.h       # GPU-resident QP pool (analogous to NvmeQueueGroup)
│   ├── rdma_target_handle.h     # Device-resident RDMA target handle
│   └── rdma_device_submit.cuh   # __device__ submit_one for RDMA
├── src/
│   ├── rdma_backend.cpp
│   ├── rdma_queue_group.cu      # ibv_create_qp + GPU MR mapping
│   └── rdma_service.cpp         # Optional daemon (like NVMeService)
└── CMakeLists.txt
```

### 4.3 RDMA Queue Group

```cpp
// backends/local_rdma/include/rdma_queue_group.h

class RdmaQueueGroup {
public:
    /// Create N RDMA QP pairs, map their SQ/CQ rings into GPU memory
    /// via ibv_reg_mr + GPUDirect RDMA (nvidia_p2p_get_pages or
    /// dma-buf for multi-vendor).
    RdmaQueueGroup(ibv_context* ctx, int cuda_device,
                   uint32_t num_qps, uint32_t queue_depth);
    ~RdmaQueueGroup();

    /// GPU-resident QP descriptor array. Kernels read SQ/CQ ring
    /// pointers from here and write Work Requests directly.
    RdmaQueuePair* d_qps() const;
    uint32_t       n_qps() const;
};
```

### 4.4 GPU-Side RDMA Submit

```cpp
// backends/local_rdma/include/rdma_device_submit.cuh

/// Submit one NVMe-oF RDMA read/write. The GPU constructs an NVMe
/// command capsule, wraps it in an RDMA Work Request, writes it into
/// the QP's SQ ring, and rings the doorbell.
__device__ __forceinline__
void rdma_submit_one(const RdmaTargetHandle* target,
                     uint64_t prp1, uint64_t prp2,
                     uint64_t remote_offset, uint64_t nbytes,
                     bool is_read)
{
    // 1. Build NVMe-oF command capsule (in GPU shared memory)
    NvmeOfCapsule* cap = build_capsule(target, remote_offset, nbytes,
                                       prp1, prp2, is_read);

    // 2. Build RDMA Work Request (send the capsule to the NVMe-oF target)
    RdmaWR wr = build_send_wr(cap, sizeof(*cap));

    // 3. Write WR into QP SQ ring, advance tail, ring doorbell
    uint32_t qidx = RdmaQueueAcquireHelper::acquire_queue(target->num_qps);
    RdmaQueuePair* qp = &target->d_qps[qidx];
    rdma_post_send(qp, &wr);

    // 4. Busy-poll CQ for completion
    RdmaQueueAcquireHelper::poll(qp, wr.id);
}
```

### 4.5 GPU-Direct RDMA Prerequisites

For GPU kernels to directly submit RDMA Work Requests:

1. **GPUDirect RDMA enabled**: The RDMA NIC and GPU must support
   peer-to-peer DMA (NVIDIA + Mellanox CX-6/CX-7 with RoCE; AMD +
   Slingshot; etc.)

2. **QP rings in GPU memory**: The QP's SQ/CQ rings must be mapped
   into GPU-visible memory. This requires:
   - `ibv_reg_mr` on GPU-allocated memory (via GPUDirect RDMA)
   - The NIC must support writing completions to GPU memory

3. **Doorbell in GPU-accessible MMIO**: The NIC's doorbell register
   must be mappable to GPU MMIO space. Some NICs (CX-7 with BlueField)
   support this; others require CPU-assisted doorbell (COOP mode).

4. **IOMMU = passthrough** (same as local NVMe path).

### 4.6 NVMe-oF Target Options

| Target type | Where it runs | Use case |
|-------------|---------------|----------|
| Remote NVMe-oF | Separate storage server | Disaggregated storage |
| SPDK NVMe-oF target | User-space on storage host | High-performance target |

## 5. Raw NVMe Access (No File)

### 5.1 Motivation

For block-oriented workloads (databases, raw KV stores, ML checkpoint
arrays), the file abstraction is unnecessary overhead. The user knows
exactly which LBA range they want to read/write.

### 5.2 API

```cpp
// Application code for raw NVMe access:
Coordinator c;
c.bootstrap(cfg);

// No file creation — direct LBA range
StorageTarget raw_target;
raw_target.kind = StorageTargetKind::NVME_RAW;
raw_target.backend_type = BackendType::LOCAL_NVME;
raw_target.device_id = 0;
raw_target.nvme_raw.namespace_id = 1;
raw_target.nvme_raw.start_lba = 0;
raw_target.nvme_raw.length_blocks = 1024 * 1024;  // 4 GiB at 4K blocks
raw_target.nvme_raw.block_size = 4096;

// Acquire a device handle for this raw target
auto* handle = c.acquire_raw_target(raw_target, stream);

// Submit IO directly against the LBA range
c.submit_batch_raw(tensor_region, handle, byte_offset, is_read, stream);
```

### 5.3 Implementation

The `local_nvme` backend's `acquire_target_handle()` for `NVME_RAW`
builds a simplified `NvmeFileDeviceHandle` with a single extent covering
the raw LBA range — no FIEMAP, no file, no persistent log. The device-side
`resolve_lba()` works identically (it walks extents regardless of how
they were created).

### 5.4 No-FIEMAP Path

When `StorageTargetKind = NVME_RAW`, the `raw_passthrough/` filesystem
short-circuits: it returns the `StorageTarget` as-is without touching
the filesystem. This means:
- No `fallocate` (the LBA range already exists on the namespace)
- No FIEMAP (the caller provides the LBA range directly)
- No `PersistentFileLog` (nothing to persist — the range is fixed)
- No `.refs/` hardlinks (no inode to protect)

## 6. Migration Path

### Phase 1: Extract filesystems/ from nvme_storage/

Move FIEMAP logic from `nvme_storage/` into `filesystems/ext4_fiemap/`:
- `fiemap_helper.h` → `filesystems/ext4_fiemap/`
- `nvme_storage` calls `IFilesystem::resolve()` instead of doing FIEMAP directly
- `NvmeFile` becomes a `StorageTarget { kind = NVME_FILE }`

### Phase 2: Add raw_passthrough/ filesystem

- `filesystems/raw_passthrough/` — no-op filesystem for raw LBA ranges
- Backend handles `NVME_RAW` targets with simplified extent setup
- No file creation path needed

### Phase 3: Add local_rdma/ backend

- Implement `IBackendProvider` for RDMA
- Implement `RdmaQueueGroup` with GPUDirect RDMA
- Implement device-side `rdma_submit_one`
- Pair with `dfs_client/` for distributed FS namespaces
- Or pair with raw passthrough for direct NVMe-oF access

### Phase 4: Unify IO engine

- Replace `NvmeBatchInputTensor` with generic
  `BufferDescriptorBatch` + `IORequestBatch` in the `IIoEngine` interface
- The IO engine becomes backend-neutral; the backend's
  `launch_batch_gpu_stream()` handles kernel dispatch
- `LocalNvmeIoEngine` becomes `LocalNvmeBackend` implementing
  `IBackendProvider`; the generic engine just dispatches

## 7. Distributed Filesystem Support (DOCA GPUNetIO)

### 7.1 The Split Data/Metadata Path Pattern

For distributed filesystems (3FS, JuiceFS, DAOS, CephFS, etc.), the
metadata and data paths are naturally split:

```
                          Application
                              │
                   ┌──────────┴──────────┐
                   │                     │
            Metadata Path            Data Path
             (CPU-side)             (GPU-side)
                   │                     │
     ┌─────────────┴─────────────┐       │
     │  DFS Client (CPU)         │       │
     │  resolve("file.txt")      │       │
     │    → block_0 @ remote_A   │       │
     │    → block_1 @ remote_B   │       │
     │    → StorageTarget[]      │       │
     └─────────────┬─────────────┘       │
                   │                     │
           StorageTarget {               │
             kind = RDMA_REMOTE,         │
             remote_addr, rkey           │
           }                             │
                   │                     │
                   └────────┬────────────┘
                            │
                    ┌───────┴────────┐
                    │ IBackendProvider│
                    │  (local_rdma)  │
                    │                │
                    │ GPU kernel:    │
                    │  construct WR  │
                    │  submit via    │
                    │  DOCA GPUNetIO │
                    │  or ibverbs    │
                    └───────┬────────┘
                            │
                     RDMA NIC (GPU-direct)
                            │
                    Remote Storage Node
```

- **Metadata** (CPU): The distributed FS client library runs on the CPU.
  It resolves file names to storage targets (remote block addresses,
  object keys, etc.). This is standard DFS client behavior — no GPU
  involvement needed.
- **Data** (GPU): The GPU directly sends/receives data over the network
  using DOCA GPUNetIO or raw RDMA verbs. The GPU kernel constructs work
  requests and submits them via NIC queues mapped to GPU memory.

### 7.2 How This Fits the SPI

The two-axis design (filesystems × backends) maps directly to this
pattern:

| Layer | Responsibility | Example |
|-------|---------------|---------|
| `filesystems/dfs_client/` | Resolve file name → `StorageTarget` (metadata, CPU-side) | 3FS client, JuiceFS client, DAOS client |
| `backends/local_rdma/` | GPU-direct data transfer via RDMA/DOCA (data, GPU-side) | ibverbs + GPUDirect RDMA, or DOCA GPUNetIO |

The `IFilesystem::resolve()` call runs on the CPU and returns a
`StorageTarget { kind = RDMA_REMOTE, rdma_remote = { remote_addr, rkey } }`.
The `IBackendProvider::launch_batch_gpu_stream()` call runs the GPU
kernel that uses this target to perform the actual data transfer.

**Key insight**: The filesystem layer never touches the data path. The
backend layer never touches the metadata path. They meet only through
`StorageTarget`.

### 7.3 DOCA GPUNetIO as a Backend Implementation

DOCA GPUNetIO is NVIDIA's high-level API for GPU-direct networking. It
provides:

1. **GPU-accessible NIC queues**: RX/TX ring buffers mapped to GPU
   memory, so GPU kernels can read incoming packets and write outgoing
   packets directly.
2. **GPU kernel networking primitives**: Send/receive/flush operations
   callable from `__device__` code, without CPU involvement.
3. **RDMA acceleration**: GPU kernels can post RDMA Work Requests
   directly to NIC hardware.

A `backends/local_rdma/` implementation using DOCA GPUNetIO would:

```cpp
class DocaRdmaBackend : public IBackendProvider {
    // ...
    void launch_batch_gpu_stream(
        IQueue* queue, AccelStream stream,
        const BufferDescriptorBatch& descs,
        const IORequestBatch& requests, bool is_read) override
    {
        // Launch GPU kernel that uses DOCA GPUNetIO
        doca_rdma_xfer_kernel<<<grid, block, 0, stream>>>(
            descs.descs, requests.requests, requests.count, is_read,
            doca_gpu_qps_);  // GPU-resident DOCA QP descriptors
    }
};
```

```cuda
// Device-side kernel (simplified):
__global__ void doca_rdma_xfer_kernel(
    const BufferDescriptor* descs, const IORequest* reqs,
    uint32_t count, bool is_read, DocaGpuQp* qps)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    const auto& req = reqs[tid];
    const auto& desc = *req.descriptor;

    // DOCA GPUNetIO: GPU directly sends RDMA read/write
    doca_gpu_rdma_send(qps[tid % NUM_QPS],
                       desc.rdma.remote_addr,
                       desc.rdma.rkey,
                       req.file_offset,  // or remote_offset
                       req.size,
                       is_read ? DOCA_RDMA_READ : DOCA_RDMA_WRITE,
                       desc.rdma.local_addr);

    // Busy-poll completion via DOCA GPUNetIO
    doca_gpu_rdma_poll(qps[tid % NUM_QPS], tid);
}
```

### 7.4 DOCA GPUNetIO vs. Raw ibverbs

| Aspect | Raw ibverbs | DOCA GPUNetIO |
|--------|-------------|---------------|
| Abstraction level | Low-level RDMA verbs | High-level GPU networking |
| GPU doorbell | Requires NIC support or COOP mode | Built-in GPU kernel API |
| Vendor lock-in | Any RDMA NIC with GPUDirect RDMA | NVIDIA-only (BlueField / CX-7) |
| Kernel module | None (stock kernel) | None (stock kernel) |
| Complexity | Manual QP/MR/CQ management | Simplified GPU API |

Both approaches implement the same `IBackendProvider` SPI — they are
 interchangeable from the runtime's perspective. A deployment can choose
 based on available hardware:

- NVIDIA + BlueField/CX-7 → DOCA GPUNetIO backend
- Mixed vendor + RDMA NIC → raw ibverbs backend
- Both use the same `StorageTarget { RDMA_REMOTE }` from the filesystem
  layer

### 7.5 Distributed FS Client Implementations

Each distributed filesystem needs its own `IFilesystem` implementation
under `filesystems/dfs_client/`:

| FS | Client library | Metadata resolution | StorageTarget kind |
|----|---------------|--------------------|--------------------|
| 3FS | 3FS client API | file → (chunk_id, node_addr) | `RDMA_REMOTE` |
| JuiceFS | JuiceFS SDK | file → (object_key, s3_addr) | `OBJECT_KEY` (future) or `RDMA_REMOTE` |
| DAOS | libdaos | object → (dkey, akey, target_rank) | `RDMA_REMOTE` |
| CephFS | libcephfs | file → (inode, extent, osd_addr) | `RDMA_REMOTE` |

The DFS client runs entirely on the CPU. It calls the DFS library to
resolve metadata, then hands the `StorageTarget` to the runtime. The
GPU backend performs the actual data transfer using the target's
remote address and memory key.

### 7.6 Practical Example: 3FS + DOCA GPUNetIO

```
Application: "read layer 5 of block 42 from 3FS"

1. CPU: filesystems/dfs_client/threefs/
   IFilesystem::resolve("block_42_layer_5")
     → 3FS client API: lookup("block_42_layer_5")
     → returns: chunk_id=123, node=10.0.0.5:8000, offset=40960
     → StorageTarget { kind=RDMA_REMOTE, remote_addr=0x..., rkey=0x... }

2. CPU: Coordinator::acquire_target_handle(target)
     → RDMA backend registers the remote target

3. GPU: IBackendProvider::launch_batch_gpu_stream(...)
     → DOCA GPUNetIO kernel: GPU sends RDMA READ from 10.0.0.5
     → Data lands directly in GPU memory (no CPU copy)
     → GPU polls for completion
```

The metadata lookup (step 1) is a standard 3FS client RPC — no GPU
involvement. The data transfer (step 3) is entirely GPU-direct — no CPU
involvement. This is the "tutti" pattern: each side plays its own
instrument.
