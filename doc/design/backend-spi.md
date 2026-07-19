# Backend SPI Design

> **Status**: v0.1. The SPI interfaces are defined and compiled; the
> runtime does not yet route through them end-to-end (the current
> `LocalNvmeIoEngine` bypasses `IBackendProvider` and calls NVMe-native
> batch types directly). This document describes the SPI contract as
> designed, with notes on current implementation gaps.

## 1. Purpose

The Backend SPI (Service Provider Interface) is the single C++ interface
the runtime core talks to. Every backend (local_nvme, RDMA, GDS, ...)
provides one `IBackendProvider` implementation. The runtime never talks
to backends directly — it goes through `device_manager` for fleet info
and through the SPI for IO.

**Design principles:**

1. **Two extension axes, independently growable:**
   - *Transport / data-path* (how bytes physically move) → `backends/`
   - *Namespace / metadata* (how a name resolves to an address) → `filesystems/`
   - They meet only through the `StorageTarget` value type.

2. **GPU kernels are not polymorphic.** There is no vtable on GPU. The
   host picks the backend and calls `launch_batch_gpu_stream()`; the
   backend's host-side method launches its own `__global__` kernel
   directly.

3. **Memory comes from the application.** Backends never allocate
   batches or buffers. The application allocates; the Memory Layer
   registers; the backend consumes `MemoryRegion*` handles.

4. **Nouns vs Services.** Runtime nouns (`Device`, `Lease`,
   `IOBuffer`, `BatchRequest`, `StorageTarget`) live in `coordinator/`
   (the `runtime/` directory referenced in earlier drafts of this
   document has since been deleted — see
   [restructuring-plan.md](../refactor/restructuring-plan.md), Decisions
   Summary). Services that produce them (`IDeviceRegistry`,
   `ILeaseManager`) live in `device_manager/`. The SPI sits between them.

## 2. Type Inventory

### 2.1 BackendType (`backend_type.h`)

```cpp
enum class BackendType : uint32_t {
    LOCAL_NVME = 0,
    RDMA       = 1,
    GDS        = 2,
    // New backends append here. Existing values must NEVER be reordered.
};
```

Wire-format consumers (`BufferDescriptor`) rely on numeric stability.
New backends append at the end.

### 2.2 IOSubmitMode (`io_submit_mode.h`)

```cpp
enum class IOSubmitMode : uint32_t {
    BATCH_GPU_STREAM = 0,  // CPU stages; GPU kernel submits via doorbell
    BATCH_CPU_SYNC   = 1,  // CPU prepares + submits + blocks
    BATCH_CPU_ASYNC  = 2,  // CPU submits, returns IOFuture
    COOP             = 3,  // Shared SQ/CQ; direction per channel
    BATCH_GPU_ASYNC  = 4,  // Persistent kernel / CUDA Graphs (out of scope v0.1)
};
```

v0.1 backends MUST implement modes 0 and 1. Mode 2 is OPTIONAL (may
return nullptr). Mode 3 is OPTIONAL (interface stable, implementation
deferred). Mode 4 is out of scope.

### 2.3 BufferDescriptor (`buffer_descriptor.h`)

Tagged union for per-slice IO metadata. The union payload MUST fit in
`raw[48]` to keep descriptor arrays cacheline-friendly on GPU.

```cpp
struct BufferDescriptor {
    BackendType backend_type;     // discriminator
    uint32_t    reserved;
    union {
        NVMeBufferDesc nvme;      // 32 bytes: kind + length + offset + PRP/SGL
        RDMABufferDesc rdma;      // 32 bytes: remote/local addr + rkey/lkey
        uint8_t raw[48];          // 48-byte budget for future backends
    };
};
```

**NVMeBufferDesc** carries the transfer kind (PRP_SINGLE / PRP_DUAL /
PRP_LIST / SGL_*), data length, tensor offset, and the PRP or SGL
addressing payload.

**RDMABufferDesc** carries remote/local addresses and memory keys.

**Invariants:**
- `backend_type` is the discriminator; readers MUST check it before
  accessing a specific union member.
- Upper layers treat the union as opaque — they pass it through but
  never read backend-specific fields.
- Adding a new backend struct larger than 48 bytes is a versioning event.

### 2.4 IORequest / IORequestBatch / IOCompletion (`io_request.h`)

```cpp
struct IORequest {
    uint64_t file_offset;                 // bytes (or remote VA)
    uint32_t size;                        // transfer size in bytes
    bool     is_read;                     // direction
    uint32_t reserved;
    const BufferDescriptor* descriptor;   // points into a descriptor pool
};

struct IORequestBatch {
    const IORequest*    requests;
    uint32_t            count;
    const MemoryRegion* region;   // app-registered home of `requests`
};

struct IOCompletion {
    uint32_t request_index;
    uint32_t status;          // 0 = success
    uint64_t bytes_done;
};
```

**Residency policy**: `IORequest` is a plain POD, valid in host memory,
device memory, pinned-shared memory, or copied between them. The struct
carries NO residency tag — residency is a property of the `MemoryRegion`
backing the array.

**Address-space semantics for `file_offset`**:
- Local NVMe: bytes from the start of the target file.
- RDMA: may carry a remote virtual address; the backend documents its
  own interpretation.

### 2.5 IQueue / QueueDesc / QueueCap (`queue.h`)

```cpp
enum QueueCap : uint32_t {
    QUEUE_CAP_CPU_ENQUEUE = 1u << 0,
    QUEUE_CAP_GPU_ENQUEUE = 1u << 1,
    QUEUE_CAP_CPU_POLL    = 1u << 2,
    QUEUE_CAP_GPU_POLL    = 1u << 3,
    QUEUE_CAP_SHARED_MEM  = 1u << 4,   // prerequisite for COOP
};

struct QueueRingDesc {
    void*    cpu_ptr;     // CPU-accessible ring base (may be nullptr)
    void*    dev_ptr;     // GPU-accessible ring base (may be nullptr)
    uint32_t depth;
    uint32_t entry_sz;
};

struct QueueDesc {
    QueueRingDesc sq;
    QueueRingDesc cq;
    uint32_t      queue_index;
    uint32_t      capabilities;
};
```

Each ring (SQ or CQ) exposes BOTH a CPU-accessible pointer and a
GPU-accessible pointer. Either may be nullptr if that side isn't
supported (e.g. a CPU-only queue sets `dev_ptr = nullptr`; NVMe with
GPUDirect sets both).

**GPU dispatch**: GPU code does NOT call `IQueue` methods (no vtable on
GPU). It reads `desc().sq.dev_ptr` / `desc().cq.dev_ptr` directly and
calls backend-private device-side helpers.

### 2.6 IQueueProvider (`queue_provider.h`)

```cpp
struct QueueConfig {
    uint32_t sq_depth;
    uint32_t cq_depth;
    uint32_t capabilities;   // required QueueCap bitmask
    int      device_id;      // CUDA device for GPU-visible alloc; -1 = CPU only
};

class IQueueProvider {
public:
    virtual IQueue* create_queue(const QueueConfig& config) = 0;
    virtual void    destroy_queue(IQueue* queue) = 0;
    virtual uint32_t active_count() const = 0;
};
```

Decoupled from `IBackendProvider` so the same NVMe backend can run with
libnvm-backed queues (default), CPU-only queues, or mock queues (tests)
— injected at construction time.

### 2.7 IOFuture (`io_future.h`)

Returned by `submit_batch_cpu_async()`. Caller owns the pointer and
MUST `delete` it after use. Backends that don't support async return
nullptr; caller falls back to `BATCH_CPU_SYNC`.

### 2.8 CoopIOChannel (`coop_channel.h`)

```cpp
struct CoopSubmitEntry {
    IORequest request;
    uint64_t  descriptor_idx;
    uint32_t  ticket;       // monotonic ID; CQ entry echoes this
    uint32_t  reserved;
};

struct CoopCompletionEntry {
    uint32_t ticket;
    uint32_t status;
    uint64_t bytes_done;
};

enum class CoopDirection : uint32_t {
    GPU_PRODUCE_CPU_SUBMIT = 0,   // GPUfs / FlashNeuron-style
    CPU_PRODUCE_GPU_SUBMIT = 1,   // CPU planner, GPU submits
};

struct CoopIOChannel {
    IQueue*       queue;    // must have QUEUE_CAP_SHARED_MEM
    CoopDirection direction;
};
```

A `CoopIOChannel` wraps an `IQueue` with `QUEUE_CAP_SHARED_MEM` set:
SQ and CQ memory is visible from both CPU and GPU simultaneously.
Producer/consumer roles are NOT fixed by the backend — `CoopDirection`
picks who produces requests and who submits to the device.

**Atomic counter contract** (left to the backend / queue provider):
- SQ head/tail counters MUST live in pinned memory accessible to both
  sides, using `cuda::atomic<uint32_t, thread_scope_system>`.
- Producer advances tail AFTER writing the entry (release).
- Consumer reads `ring[head % depth]`, processes, then advances head
  (release).

## 3. IBackendProvider Interface

```cpp
class IBackendProvider {
public:
    // 4.1 Memory Layer interface
    virtual bool prepare_descriptors(
        const uint64_t*      ioaddrs,
        std::size_t          n_ioaddrs,
        const SubSliceInfo*  slices,
        std::size_t          n_slices,
        BufferDescriptor*    out_descs) = 0;

    // 4.2 Device Manager interface
    virtual IQueue* acquire_queue(uint32_t required_caps) = 0;
    virtual void    release_queue(IQueue* queue) = 0;
    virtual IQueueProvider* queue_provider() = 0;

    // 4.3 Lifecycle
    virtual bool initialize() = 0;
    virtual void cleanup() = 0;

    // 4.4 IO Engine interface (four submission modes)
    virtual void launch_batch_gpu_stream(
        IQueue*, cudaStream_t,
        const BufferDescriptorBatch&,
        const IORequestBatch&, bool is_read) = 0;

    virtual bool submit_batch_cpu_sync(
        IQueue*,
        const BufferDescriptorBatch&,
        const IORequestBatch&, bool is_read) = 0;

    virtual IOFuture* submit_batch_cpu_async(...) = 0;  // OPTIONAL
    virtual bool setup_coop_channel(...) = 0;           // OPTIONAL
    virtual uint32_t drain_coop_channel(...) = 0;       // OPTIONAL
    virtual void teardown_coop_channel(...) = 0;        // OPTIONAL

    // 4.5 Metadata
    virtual BackendType  backend_type() const = 0;
    virtual const char*  backend_name() const = 0;
    virtual std::size_t  max_io_size() const = 0;
    virtual std::size_t  queue_depth() const = 0;
    virtual std::size_t  queue_count() const = 0;
};
```

### 3.1 v0.1 Implementation Requirements

| Method | Required? | local_nvme status |
|--------|-----------|-------------------|
| `prepare_descriptors` | REQUIRED | Not yet wired (memory layer builds PRPs directly) |
| `acquire_queue` / `release_queue` | REQUIRED | Not yet wired (NvmeQueueGroup used directly) |
| `queue_provider` | REQUIRED | Not yet wired |
| `initialize` / `cleanup` | REQUIRED | Not yet wired |
| `launch_batch_gpu_stream` | REQUIRED | Not yet wired (LocalNvmeIoEngine calls NVMe-native kernel) |
| `submit_batch_cpu_sync` | REQUIRED | Not yet wired |
| `submit_batch_cpu_async` | OPTIONAL | Not implemented |
| `setup/drain/teardown_coop_channel` | OPTIONAL | Not implemented |

> **Current gap**: The interfaces are defined but the runtime does not
> route through them. `LocalNvmeIoEngine` directly uses
> `NvmeBatchInputTensor` / `NvmeBatchEntry` and calls the NVMe-native
> `nvme_batch_xfer_kernel`. Wiring the SPI is a Phase 4 deliverable.

### 3.2 GPU Dispatch Model

GPU kernels are NOT polymorphic. The host picks the backend and calls
`launch_batch_gpu_stream()`; the backend's host-side method launches its
OWN `__global__` kernel directly.

This means:
- Each backend ships its own `.cu` / `.cuh` files.
- The runtime core never includes backend-specific kernel headers.
- Backend selection is a host-side decision; the GPU binary is fixed at
  link time (or loaded via CUDA module / dynamic parallelism in future).

**Example** (local_nvme):
```cpp
void LocalNvmeBackend::launch_batch_gpu_stream(
    IQueue* queue, cudaStream_t stream,
    const BufferDescriptorBatch& descs,
    const IORequestBatch& requests, bool is_read)
{
    // Backend launches its OWN kernel — no virtual dispatch on GPU
    nvme_batch_xfer_kernel<<<grid, block, 0, stream>>>(
        descs.descs, requests.requests, requests.count, is_read,
        /* backend-private queue desc */ ...);
}
```

## 4. Backend Registration Flow

```
Backend Implementation
  │
  ├── IDeviceRegistry::Open()
  │     discover devices, create Device{backend_private=...}
  │     register into registry
  │
  ├── IBackendProvider::initialize()
  │     called after queues + DMA contexts ready
  │     finalize device-side init (register pools, populate constant memory)
  │
  └── Runtime now holds:
        Device.backend_private  → backend-specific state
        IBackendProvider*       → for IO submission
        IQueueProvider*         → for queue factory
```

The runtime never talks to backends directly:
- **Fleet info** → through `device_manager` (`IDeviceRegistry`)
- **IO** → through the SPI (`IBackendProvider`)

## 5. Filesystem / Backend Composition

A `Device` registered into the runtime carries both a backend provider
(data-path) and an associated filesystem resolver (namespace). They are
paired at config time, not built in. This makes new combinations cheap:

| Use case | filesystems/ | backends/ |
|---|---|---|
| Local NVMe + on-device file layout | `tutti_layout/` | `local_nvme/` |
| Local NVMe + ext4 files | `ext4_fiemap/` | `local_nvme/` |
| Local RDMA + distributed FS client | `dfs_client/<x>/` | `local_rdma/` |
| GDS + ext4 files | `ext4_fiemap/` | `gds/` |
| Local NVMe raw (no FS) | none (passthrough `BLOCK_RANGE`) | `local_nvme/` |

The filesystem layer's only job is to produce a `StorageTarget`; the
backend's only job is to consume one. Neither layer includes the other's
private headers.

> **Current state**: The `filesystems/` layer does not yet exist.
  FIEMAP extent extraction is baked into `nvme_storage`. Extracting it
  is a Phase 2+ deliverable.

## 6. StorageTarget (Planned)

The `StorageTarget` value type is the meeting point between filesystems
and backends. It carries the addressing payload a backend needs to
reach the data, without exposing filesystem internals.

```cpp
// Planned shape (not yet implemented):
struct StorageTarget {
    BackendType  backend_type;
    uint32_t     device_id;
    // Tagged union — backend-specific addressing
    union {
        NvmeFileTarget   nvme_file;     // (file_id, LBA extent list)
        NvmeRawTarget    nvme_raw;      // (namespace_id, LBA range)
        RdmaTarget       rdma;          // (remote_addr, rkey, length)
        // Future: object key, DFS handle, ...
    };
};
```

The filesystem layer resolves a name (file path, object key, DFS handle)
into a `StorageTarget`. The backend consumes the `StorageTarget` to
build its `BufferDescriptor` / `IORequest` batches. Adding one new
transport does not require touching any filesystem code, and vice versa.

## 7. KV Cache Workload Analysis (vLLM / LMCache)

The primary upper-layer use case for Tutti is LLM KV-cache offloading:
when GPU memory is full, KV-cache blocks are swapped to NVMe storage and
read back on demand. This section analyzes whether the SPI supports this
pattern and identifies gaps.

### 7.1 The KV Cache IO Pattern

In vLLM and LMCache, the KV cache is organized as:

```
Block 0:  Layer 0: [K][V]  Layer 1: [K][V]  ...  Layer N: [K][V]
Block 1:  Layer 0: [K][V]  Layer 1: [K][V]  ...  Layer N: [K][V]
...
Block M:  Layer 0: [K][V]  Layer 1: [K][V]  ...  Layer N: [K][V]
```

Each `[K]` or `[V]` is one tensor of `tensor_size` bytes. One GpuFile
stores all layers for one block, sharded across NVMe devices. The IO
pattern at decode time:

```
For a batch of blocks [b0, b1, ..., bM] at layer L:
  Read K[b0][L], V[b0][L], K[b1][L], V[b1][L], ..., K[bM][L], V[bM][L]
```

This is a **batch of uniform-direction IOs** against multiple files,
where each IO targets a different file offset within the same layer.

### 7.2 SPI Mapping

| KV cache need | SPI support | How |
|---------------|-------------|-----|
| Batch of many tensors, uniform direction (all read or all write) | Supported | `IORequestBatch { requests, count, region }` + `is_read` parameter |
| Per-tensor file offset | Supported | `IORequest::file_offset` |
| Per-tensor buffer descriptor (PRP/RDMA addr) | Supported | `IORequest::descriptor` points into `BufferDescriptorBatch` |
| Many files in one batch | Supported | Each `IORequest` can reference a different `StorageTarget` handle via its descriptor |
| Auto-chunking when batch exceeds capacity | Adapter-level | `KvCacheIoAdapter` greedily packs under `max_entries_per_batch` |
| Handle caching (file_id -> device handle) | Supported | `Coordinator::handle_for_batch()` resolves and caches |
| Memory registration with per-tensor granularity | Supported | `register_tensor(granularity = tensor_size)` pre-computes IO-slice table |
| K/V offset arithmetic | Adapter-level | `KvCacheIoAdapter` computes `layer * tensor_size * 2` (standard) or `layer * tensor_size` (MLA) |
| Layer-level stream overlap | Supported | One `AccelStream` per layer; one adapter/engine per stream |

### 7.3 vLLM Integration

vLLM manages its own block table and KV cache memory pool. Integration:

```cpp
// vLLM-side (simplified):
class TuttiKVCacheManager {
    Coordinator coord_;
    KvCacheIoAdapter adapter_;
    std::vector<MemoryRegion*> kv_tensor_regions_;  // one per (block, layer, K/V)

    void swap_out(int layer, std::vector<int>& block_ids) {
        // K and V tensors for each block at this layer
        std::vector<MemoryRegion*> k_regions, v_regions;
        std::vector<GpuFileId> file_ids;
        for (int b : block_ids) {
            k_regions.push_back(kv_tensor_regions_[k_idx(b, layer)]);
            v_regions.push_back(kv_tensor_regions_[v_idx(b, layer)]);
            file_ids.push_back(block_file_id_[b]);
        }
        adapter_.batched_write(layer, k_regions, v_regions,
                               file_ids, stream_);
    }

    void swap_in(int layer, std::vector<int>& block_ids) {
        // Same but batched_read
        adapter_.batched_read(layer, k_regions, v_regions,
                              file_ids, stream_);
    }
};
```

**What Tutti provides**: batch IO submission, GPU-direct NVMe access,
handle caching, PRP pre-computation.
**What vLLM keeps**: block table, eviction policy, KV cache memory pool,
layer scheduling.

### 7.4 LMCache Integration

LMCache is a distributed KV-cache layer that caches across nodes. It
needs persistent storage for "warm" cache pages. Integration:

```cpp
// LMCache-side (simplified):
class TuttiLMCacheStorage {
    Coordinator coord_;
    KvCacheIoAdapter adapter_;

    // LMCache manages its own LRU index of (token_hash -> GpuFileId)
    std::unordered_map<TokenHash, GpuFileId> cache_index_;

    void store(TokenHash hash, int layer,
               MemoryRegion* k, MemoryRegion* v) {
        GpuFileId fid = get_or_create_file(hash);
        adapter_.batched_write(layer, {k}, {v}, {fid}, stream_);
    }

    bool lookup(TokenHash hash, int layer,
                MemoryRegion* k, MemoryRegion* v) {
        auto it = cache_index_.find(hash);
        if (it == cache_index_.end()) return false;
        adapter_.batched_read(layer, {k}, {v}, {it->second}, stream_);
        return true;
    }
};
```

**What Tutti provides**: persistent GpuFile lifecycle, batch IO, metadata
persistence (`PersistentFileLog` / `gpu_file_log.bin`).
**What LMCache keeps**: cache index, token hashing, cross-node
coordination, LRU policy.

### 7.5 Current Gaps

| Gap | Impact | Resolution |
|-----|--------|------------|
| `KvCacheIoAdapter` uses `NvmeBatchInputTensor` (NVMe-specific) | Adapter only works with local_nvme backend | Phase 4: replace with generic `BufferDescriptorBatch` + `IORequestBatch` |
| `cudaStream_t` in adapter header | CUDA-bound | Phase 1: replace with `AccelStream` |
| No multi-layer batch in one submit | Each layer is a separate `submit_batch` call (blocks until done) | Future: `submit_batch_async` for stream overlap across layers |
| One `MemoryRegion` per (block, layer, K/V) | Many small registrations; no slice-scoped addressing | Future: register whole per-layer cache once, address slices via `lookup_io_slice` |
| No priority/scheduling | All IOs in a batch are equal priority | Future: per-request priority in `IORequest::reserved` |

### 7.6 Conclusion

The SPI design **fully supports** the KV-cache read/write pattern for
vLLM and LMCache. The core abstractions (`IORequestBatch`,
`BufferDescriptorBatch`, `IBackendProvider::launch_batch_gpu_stream`)
map directly to the batch-of-tensors-per-layer IO pattern. The
`KvCacheIoAdapter` already demonstrates this works end-to-end.

The main gap is that the adapter is currently NVMe-specific — it uses
`NvmeBatchInputTensor` instead of the generic SPI types. Phase 4
(generic `IIoEngine`) resolves this, making the adapter backend-neutral
so the same KV-cache code works with `local_nvme`, `gds`, and future
`local_rdma` backends.
