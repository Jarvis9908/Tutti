# Accelerator Abstraction Design

> **Status**: Design proposal. Not yet implemented. This document
> addresses the requirement to decouple Tutti from CUDA/NVIDIA to
> support different GPU and NPU vendors.

## 1. Problem Statement

The current codebase is tightly coupled to NVIDIA CUDA. This prevents
porting to other accelerators (AMD ROCm, Intel oneAPI/SYCL, Huawei
Ascend CANN, generic NPU) and limits adoption in heterogeneous
environments.

### 1.1 Coupling Inventory

The following CUDA dependencies exist in **public interface headers**
(should not depend on a specific accelerator):

| File | CUDA dependency | Impact |
|------|-----------------|--------|
| `coordinator/include/coordinator.h` | `cuda_runtime.h`, `cudaStream_t` in 8+ methods | Entire top-level API is CUDA-bound |
| `io_engine/include/io_engine.h` | `cuda_runtime.h`, `cudaStream_t` | IIoEngine interface CUDA-bound |
| `io_engine/include/backend_provider.h` | `cuda_runtime.h`, `cudaStream_t` | Backend SPI CUDA-bound |
| `memory/include/memory_subsystem.h` | `cuda_runtime.h`, `cudaStream_t` | Memory subsystem CUDA-bound |
| `memory/include/host_device_memory_subsystem.h` | `nvm_types.h` (nvm_dma_t) | Memory impl leaks libnvm |
| `device_manager/include/nvme_queue_group.h` | `nvm_types.h`, QueuePair (libnvm) | Queue group CUDA-bound via libnvm |
| `device_manager/include/local_nvme_device.h` | `nvm_types.h` (nvm_ctrl_t) | Device payload leaks libnvm |

The following CUDA dependencies exist in **implementation files** (can
be isolated behind an abstraction):

| Area | CUDA calls |
|------|------------|
| Memory allocation | `cudaMalloc`, `cudaFree`, `cudaMallocHost`, `cudaFreeHost`, `cudaMallocManaged` |
| Memory registration | `cudaHostRegister`, `cudaHostGetDevicePointer`, `cudaIpcOpenMemHandle` |
| DMA mapping | `nvm_dma_map_data_device` (internally calls `cudaGetDevicePointer` / GPUDirect RDMA) |
| Stream / sync | `cudaStreamSynchronize`, `cudaStreamCreate`, `cudaGetLastError` |
| Data transfer | `cudaMemcpyAsync` |
| Queue group | `cudaMalloc`, `cudaMemcpy`, `cudaHostGetDevicePointer`, `cudaFree` |
| Kernel launch | `<<<grid, block, 0, stream>>>` syntax |
| Device code | `__device__`, `__global__`, `__forceinline__`, `cuda::atomic` |

### 1.2 Goals

1. **Public headers must be accelerator-neutral.** No `cuda_runtime.h`
   in any interface header. Replace `cudaStream_t` with an opaque handle.

2. **Implementation behind an `IAccelerator` interface.** Each vendor
   provides one implementation (CUDA, ROCm, SYCL, CANN, ...).

3. **Device-side code must be compilable for multiple targets.** Use
   macro-based or header-based dispatch (`__device__` → vendor
   equivalent).

4. **Backend DMA mapping must go through the accelerator interface.**
   `nvm_dma_map_data_device` is CUDA-specific; the abstraction provides
   a vendor-neutral DMA mapping entry point.

5. **No performance regression on the CUDA path.** The abstraction
   must be zero-cost (inline wrappers, compile-time dispatch).

## 2. Proposed Abstraction Layer

### 2.1 Layer Position

```
┌────────────────────────────────────────────────────┐
│              Coordinator / Runtime                   │
├────────────────────────────────────────────────────┤
│    io_engine    │    memory     │   device_manager   │
├────────────────────────────────────────────────────┤
│          Accelerator Abstraction Layer               │  ← NEW
│            (accel/ — IAccelerator)                   │
├─────────┬──────────┬───────────┬───────────────────┤
│  CUDA   │  ROCm    │  SYCL     │  CANN / other     │
│  impl   │  impl    │  impl     │  impl             │
├────────────────────────────────────────────────────┤
│              backends/local                          │
│         (kernel_modules, libnvm, NVMeService)        │
└────────────────────────────────────────────────────┘
```

The `accel/` layer sits between the core runtime modules and the
backends. It is the single point where accelerator-specific APIs are
called.

### 2.2 IAccelerator Interface

```cpp
// accel/include/accelerator.h

namespace tutti {

/// Opaque stream handle — replaces cudaStream_t in all public headers.
using AccelStream = void*;
using AccelEvent  = void*;

/// Memory kind (mirrors MemoryKind but adds accelerator-specific semantics).
enum class AccelMemoryLocation {
    HOST,            // plain malloc
    PINNED_HOST,     // page-locked, device-visible
    DEVICE,          // on-accelerator memory
    MANAGED,         // unified virtual addressing
    EXTERNAL,        // app-supplied
};

/// Allocation request.
struct AccelAllocSpec {
    AccelMemoryLocation location;
    std::size_t          size;
    int                  device_id;       // -1 for host
    std::size_t          alignment;       // 0 = default
};

/// DMA mapping result — vendor-neutral replacement for nvm_dma_t.
struct AccelDmaMapping {
    const uint64_t* ioaddrs;        // per-page DMA bus addresses
    std::size_t     ioaddr_count;
    std::size_t     page_size;
    void*           backend_private; // vendor-specific handle (nvm_dma_t*, ibv_mr*, ...)
};

class IAccelerator {
public:
    virtual ~IAccelerator() = default;

    // --- Identity ---
    virtual const char* name() const = 0;          // "cuda", "rocm", "sycl", ...
    virtual int         device_count() const = 0;
    virtual bool        set_device(int device_id) = 0;

    // --- Stream / Event ---
    virtual AccelStream create_stream() = 0;
    virtual void        destroy_stream(AccelStream) = 0;
    virtual void        synchronize_stream(AccelStream) = 0;
    virtual AccelEvent  create_event() = 0;
    virtual void        destroy_event(AccelEvent) = 0;
    virtual void        record_event(AccelEvent, AccelStream) = 0;
    virtual bool        query_event(AccelEvent) = 0;
    virtual void        wait_event(AccelEvent, AccelStream) = 0;

    // --- Memory allocation ---
    virtual void* allocate(const AccelAllocSpec& spec) = 0;
    virtual void  free(void* ptr, AccelMemoryLocation loc) = 0;

    // --- Memory transfer ---
    virtual bool memcpy_async(void* dst, const void* src, std::size_t size,
                              AccelStream stream,
                              AccelMemcpyKind kind) = 0;

    // --- DMA mapping (replaces nvm_dma_map_data_device) ---
    /// Map a device/host buffer for NVMe DMA. Returns per-page bus
    /// addresses usable by any NVMe controller.
    virtual bool dma_map(void* ptr, std::size_t size, int device_id,
                         AccelDmaMapping* out) = 0;
    virtual void dma_unmap(AccelDmaMapping* mapping) = 0;

    // --- Host pointer → device pointer (for pinned host memory) ---
    virtual void* get_device_ptr(void* host_ptr) = 0;

    // --- IPC (cross-process sharing) ---
    virtual bool ipc_export(void* device_ptr, AccelIpcHandle* out) = 0;
    virtual void* ipc_import(const AccelIpcHandle& handle, int device_id) = 0;
    virtual void  ipc_close(void* device_ptr) = 0;

    // --- Error handling ---
    virtual const char* last_error() const = 0;
};

} // namespace tutti
```

### 2.3 Header-Level Changes

All public interface headers replace `cudaStream_t` with `AccelStream`
(aliased as `void*`):

**Before:**
```cpp
// coordinator/include/coordinator.h
#include <cuda_runtime.h>
bool submit_batch(..., cudaStream_t stream = 0);
```

**After:**
```cpp
// coordinator/include/coordinator.h
#include "accel/include/accelerator.h"   // AccelStream = void*
bool submit_batch(..., AccelStream stream = nullptr);
```

The same pattern applies to `io_engine.h`, `backend_provider.h`, and
`memory_subsystem.h`.

Headers that already use `void*` aliases (`nvme_storage.h`'s
`GpuStreamHandle`, `block_storage.h`'s `GpuStreamHandle`) are already
accelerator-neutral in shape — they just need to be unified under
`AccelStream`.

### 2.4 Device-Side Code Abstraction

Device-side kernel code (`__device__`, `__global__`, `__forceinline__`)
needs a compilation-time dispatch mechanism. Two approaches:

#### Approach A: Macro-based (Zero-cost, compile-time)

```cpp
// accel/include/accel_device_macros.h

#if defined(TUTTI_ACCEL_CUDA)
#  define TUTTI_DEVICE        __device__
#  define TUTTI_GLOBAL        __global__
#  define TUTTI_FORCEINLINE   __forceinline__
#  define TUTTI_HOST_DEVICE   __host__ __device__
#  define TUTTI_ATOMIC_U32    cuda::atomic<uint32_t, cuda::thread_scope_system>
#  define TUTTI_LAUNCH_KERNEL(kernel, grid, block, shared, stream, ...)  \
       kernel<<<grid, block, shared, stream>>>(__VA_ARGS__)
#elif defined(TUTTI_ACCEL_ROCM)
#  define TUTTI_DEVICE        __device__
#  define TUTTI_GLOBAL        __global__
#  define TUTTI_FORCEINLINE   __forceinline__
#  define TUTTI_HOST_DEVICE   __host__ __device__
#  define TUTTI_ATOMIC_U32    hipAtomic_t<uint32_t>
#  define TUTTI_LAUNCH_KERNEL(kernel, grid, block, shared, stream, ...)  \
       hipLaunchKernelGGL(kernel, grid, block, shared, stream, __VA_ARGS__)
#elif defined(TUTTI_ACCEL_SYCL)
// SYCL uses C++ lambdas, not attributes — different compilation model
// (would require a wrapper layer, not just macros)
#elif defined(TUTTI_ACCEL_CANN)
// Ascend CANN kernel launch
#endif
```

**Pros**: Zero-cost, no abstraction overhead, each backend compiles for
its target.

**Cons**: SYCL/oneAPI have fundamentally different compilation models
(C++ lambdas vs. attributes) and may need a different approach.

### 2.5 DMA Mapping Abstraction

The current `nvm_dma_map_data_device` is CUDA-specific — it internally
calls `cuMemGetAddressRange` / GPUDirect RDMA APIs to get PCI bus
addresses. Other accelerators have equivalent mechanisms:

| Accelerator | DMA mapping API |
|-------------|-----------------|
| NVIDIA CUDA | `nvm_dma_map_data_device` (libnvm wraps GPUDirect RDMA) |
| AMD ROCm | `amdgpu_bo_get_handle` + `ibv_reg_mr` (RDMA) or direct PCIe |
| Intel oneAPI | `zeMemGetIpcHandle` + Level Zero DMA |
| Huawei CANN | `aclrtGetMemInfo` + proprietary DMA |

The `IAccelerator::dma_map()` entry point provides a vendor-neutral
wrapper. The `local_nvme` backend's kernel module (`snvme`) must also
be adapted — currently it uses NVIDIA's `nvidia_p2p_get_pages` to pin
GPU pages. For other vendors, the equivalent page-pinning mechanism
must be used (or GPUDirect RDMA must be available for that vendor).

### 2.6 Queue Ring Dual-Pointer

The `QueueRingDesc` dual-pointer design (`cpu_ptr` + `dev_ptr`) is
already accelerator-neutral. For CUDA, `dev_ptr` comes from
`cudaHostGetDevicePointer` on pinned memory. For other accelerators,
the equivalent host-visible device pointer is obtained via
`IAccelerator::get_device_ptr()`.

## 3. Migration Strategy

### Phase 1: Introduce `accel/` with CUDA implementation

- Create `accel/include/accelerator.h` with the `IAccelerator` interface.
- Create `accel/cuda/` with `CudaAccelerator` wrapping all CUDA calls.
- Replace `cudaStream_t` with `AccelStream` in all public headers.
- Replace direct `cudaMalloc`/`cudaFree`/`cudaMemcpy` calls with
  `IAccelerator` method calls in implementation files.
- Replace `nvm_dma_map_data_device` with `IAccelerator::dma_map()` in
  the memory subsystem.
- No behavior change — just routing through the abstraction.

### Phase 2: Remove `cuda_runtime.h` from public headers

- Move all `#include <cuda_runtime.h>` from `.h` to `.cpp` / `.cu`.
- Public headers include only `accel/include/accelerator.h`.
- Verify: an application that links Tutti should NOT need CUDA headers
  to use the public API (only to use the GPU submit path).

### Phase 3: Device-side macro abstraction

- Replace `__device__` / `__global__` / `__forceinline__` with
  `TUTTI_DEVICE` / `TUTTI_GLOBAL` / `TUTTI_FORCEINLINE` in all `.cuh`
  files.
- Replace `cuda::atomic` with `TUTTI_ATOMIC_U32`.
- Replace `<<<grid, block, stream>>>` with `TUTTI_LAUNCH_KERNEL`.
- Verify CUDA path still compiles and runs identically.

### Phase 4: Second backend proof-of-concept

- Implement `accel/rocm/` with `RocmAccelerator`.
- Verify the abstraction is sufficient by porting one smoke test.
- This phase validates the design; it does not need to be complete.

## 4. What Stays CUDA-Specific

Some things cannot be fully abstracted without unacceptable cost:

1. **GPUDirect RDMA / P2P page pinning**: The `snvme` kernel module
   currently calls `nvidia_p2p_get_pages`. For other vendors, a
   vendor-specific kernel module or a generic `dma_buf`-based approach
   is needed. This is a backend-level concern, not a runtime concern.

2. **Kernel binary format**: CUDA `.fatbin`, ROCm `.co`, etc. are
   incompatible. The runtime must load the correct binary for the
   active accelerator. This is handled by linking the correct
   `accel/<vendor>/` implementation.

3. **Intrinsic functions**: Warp shuffle, shared memory, cooperative
   groups — these have different syntax across vendors. Backend device
   code must be written per-vendor.
