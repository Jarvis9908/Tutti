# Project Structure Restructuring Plan

> **Status**: Confirmed. All decision points resolved.
> This document synthesizes the design documents
> ([backend-spi](../design/backend-spi.md),
> [gpu-abstraction](../design/gpu-abstraction.md),
> [kernel-portability](../design/kernel-portability.md),
> [storage-extensibility](../design/storage-extensibility.md))
> into a concrete restructuring plan.

## 1. Decisions Summary

| Decision | Resolution |
|----------|------------|
| `runtime/` directory | Deleted. Nouns stay in `coordinator/`. Duplicate `device.h` resolved. |
| `nvme_storage/` | Fully absorbed into `backends/local_nvme/`. |
| `IIoEngine` | Replaced with generic interface (clean break, no NVMe-specific types). |
| Backward compatibility | Clean break. Update all callers in `examples/` and `adapters/`. |
| Backend priority | GDS backend first. RDMA backend as community issue (framework only). |
| `libnvm` | Absorbed into `backends/local_nvme/` source (no separate subdirectory). |
| GPU abstraction | Macro/compilation approach for all vendors (CUDA/ROCm/SYCL/CANN). |
| DMA mapping for non-CUDA | Vendor-specific kernel module extensions (snvme equivalent per vendor). |
| Kernel versions (v0.1) | 5.4 (TencentOS) and 5.15 (Ubuntu) only. No 6.x yet. |

## 2. Current Problems Summary

| # | Problem | Resolution |
|---|---------|------------|
| P1 | `runtime/` was empty; duplicate `device.h` | `runtime/` deleted; `coordinator/` is canonical home for nouns |
| P2 | No `filesystems/` layer; FIEMAP baked into `nvme_storage/` | Extract `filesystems/` layer (Phase 2) |
| P3 | `IIoEngine` takes `NvmeBatchInputTensor` (NVMe-specific) | Replace with generic interface (Phase 4) |
| P4 | CUDA hardcoded in public headers | Introduce `accel/` layer (Phase 1) |
| P5 | `libnvm` types leak into `memory/` and `device_manager/` | Move types to `backends/local_nvme/` (Phase 3) |
| P6 | Two kernel module directory copies, no DKMS | Consolidate with DKMS (Phase 6) |
| P7 | File is the only storage abstraction | Introduce `StorageTarget` (Phase 5) |

## 3. Target Directory Structure

```
Tutti/
├── api/                    # Public runtime API (application-facing headers)
│   └── include/
│       ├── tutti.h         #   Umbrella header
│       ├── runtime.h       #   Runtime lifecycle
│       ├── io.h            #   IO submission API
│       ├── memory.h        #   Memory registration API
│       └── file.h          #   File/target lifecycle API
│
├── accel/                  # Accelerator abstraction layer (NEW)
│   ├── include/
│   │   ├── accelerator.h   #   IAccelerator interface
│   │   └── accel_device_macros.h  # TUTTI_DEVICE, TUTTI_GLOBAL, ...
│   ├── cuda/               #   CUDA implementation
│   │   ├── cuda_accelerator.h
│   │   └── cuda_accelerator.cpp
│   └── CMakeLists.txt
│
├── coordinator/            # Top-level orchestrator + runtime nouns
│   └── include/
│       ├── coordinator.h         #   Coordinator (remove cuda_runtime.h)
│       ├── coordinator_config.h
│       ├── device.h              #   Device (canonical home)
│       ├── capability_set.h      #   CapabilitySet
│       ├── lease.h               #   Lease
│       └── storage_target.h      #   StorageTarget (NEW)
│
├── memory/                 # Memory subsystem (unchanged location, cleaned deps)
│   └── include/
│       ├── memory_subsystem.h         # (remove cuda_runtime.h, use AccelStream)
│       ├── memory_region.h
│       ├── host_device_memory_subsystem.h  # (remove nvm_types.h)
│       └── ... (unchanged)
│
├── device_manager/         # Device fleet management (cleaned, thinned)
│   └── include/
│       ├── device_registry.h          # IDeviceRegistry (stays)
│       └── lease_manager.h            # ILeaseManager (stays)
│       # (local_nvme_device.h, nvme_queue_group.h, *_registry.h
│       #  all move to backends/local_nvme/)
│
├── io_engine/              # IO submission SPI (generic, backend-neutral)
│   └── include/
│       ├── io_engine.h          # IIoEngine (generic, no NvmeBatchInputTensor)
│       ├── backend_provider.h   # IBackendProvider
│       ├── buffer_descriptor.h
│       ├── io_request.h
│       ├── queue.h
│       ├── queue_provider.h
│       ├── io_future.h
│       ├── coop_channel.h
│       ├── io_submit_mode.h
│       └── backend_type.h
│       # (local_nvme/ subdirectory moves to backends/local_nvme/)
│
├── filesystems/            # Namespace -> StorageTarget (NEW, extracted from nvme_storage)
│   ├── include/
│   │   └── filesystem.h    #   IFilesystem interface
│   ├── ext4_fiemap/        #   ext4 + FS_IOC_FIEMAP (moved from nvme_storage)
│   ├── tutti_layout/       #   Custom on-device layout (future)
│   ├── dfs_client/         #   Distributed FS clients (future)
│   └── raw_passthrough/    #   No-op for raw NVMe LBA ranges
│
├── backends/               # Data-path backends (transport only)
│   ├── include/            #   Cross-backend helpers
│   ├── local_nvme/         #   (renamed from local/; absorbs nvme_storage + libnvm)
│   │   ├── include/
│   │   │   ├── local_nvme_backend.h        # IBackendProvider impl
│   │   │   ├── local_nvme_device.h         # (moved from device_manager/)
│   │   │   ├── nvme_queue_group.h          # (moved from device_manager/)
│   │   │   ├── local_nvme_direct_registry.h    # (moved from device_manager/)
│   │   │   ├── nvmeservice_backed_registry.h   # (moved from device_manager/)
│   │   │   ├── nvme_file_device_handle.h  # (moved from nvme_storage/)
│   │   │   ├── nvme_device_submit.cuh     # (moved from nvme_storage/)
│   │   │   ├── queue_acquire_helper.cuh   # (moved from nvme_storage/)
│   │   │   ├── nvme_file.h                # (moved from nvme_storage/)
│   │   │   ├── lba_extent.h               # (moved from nvme_storage/)
│   │   │   ├── nvme_file_header.h         # (moved from nvme_storage/)
│   │   │   ├── persistent_file_log.h      # (moved from nvme_storage/)
│   │   │   ├── host_fs_backed_nvme_storage.h  # (moved from nvme_storage/)
│   │   │   ├── host_batch_builder.h       # (moved from io_engine/local_nvme/)
│   │   │   ├── launch_batch.h             # (moved from io_engine/local_nvme/)
│   │   │   ├── local_nvme_io_engine.h     # (moved from io_engine/local_nvme/)
│   │   │   ├── nvme_batch.h               # (moved from io_engine/local_nvme/)
│   │   │   └── libnvm/                    # Absorbed libnvm headers
│   │   ├── src/
│   │   │   ├── libnvm/                    # Absorbed libnvm source
│   │   │   └── ...
│   │   ├── kernel_modules/                # snvme (DKMS, single source + patches)
│   │   │   ├── snvme/                     #   Single source tree
│   │   │   ├── patches/                   #   Version-specific patches
│   │   │   ├── dkms.conf
│   │   │   └── Makefile
│   │   ├── nvmeservice/                   # Daemon (renamed from NVMeService/)
│   │   └── CMakeLists.txt
│   └── gds/                               # GDS backend (next priority)
│
├── block_storage/          # GpuFile abstraction (stays, becomes optional layer)
│   └── include/
│       ├── block_storage.h                # (GpuFile = one StorageTarget shape)
│       └── ... (unchanged)
│
├── adapters/               # Framework integrations (unchanged)
├── examples/               # Examples and smoke tests (update for new API)
├── scripts/                # Setup scripts (unchanged)
├── doc/                    # Documentation
├── CMakeLists.txt
├── sys_config.yaml
├── Roadmap.md
└── README.md
```

### Key structural changes:

1. **`coordinator/` owns nouns** — `Device`, `CapabilitySet`, `Lease`,
   `StorageTarget` stay here. `runtime/` is deleted. No more duplicate
   `device.h` issue.

2. **`accel/` is new** — `IAccelerator` interface + CUDA implementation.
   All `cuda_runtime.h` includes move out of public headers into
   `accel/cuda/`. Device-side macros (`TUTTI_DEVICE`, `TUTTI_GLOBAL`, ...)
   support all vendors via compilation dispatch.

3. **`filesystems/` is new** — extracted from `nvme_storage/`. FIEMAP
   logic moves to `filesystems/ext4_fiemap/`.

4. **`nvme_storage/` is fully absorbed** into `backends/local_nvme/` —
   all its types move to `backends/local_nvme/include/`, its FIEMAP
   logic to `filesystems/ext4_fiemap/`. The `nvme_storage/` directory
   is deleted.

5. **`libnvm` is absorbed** into `backends/local_nvme/` source — the
   `backends/local/nvme/libnvm/` directory becomes
   `backends/local_nvme/src/libnvm/` (source) +
   `backends/local_nvme/include/libnvm/` (headers). No separate
   subdirectory.

6. **`block_storage/` stays** — but `GpuFile` becomes one
   `StorageTarget` shape (file-backed), not the only abstraction. Raw
   NVMe and RDMA targets bypass `block_storage/` entirely.

7. **`device_manager/` is thinned** — backend-specific types
   (`LocalNvmeDevice`, `NvmeQueueGroup`, `LocalNvmeDirectRegistry`,
   `NvmeServiceBackedRegistry`) move to `backends/local_nvme/`.
   `device_manager/` keeps only `IDeviceRegistry` and `ILeaseManager`.

8. **`backends/local/` -> `backends/local_nvme/`** — renamed for
   clarity. `NVMeService/` -> `nvmeservice/` (consistent casing).

9. **`kernel_modules/` consolidated** — single source tree + patch
   series + DKMS. Supports 5.4 and 5.15 for v0.1.

## 4. Phased Execution Plan

### Phase 1: Introduce `accel/` abstraction + device-side macros

**Steps**:
1. Create `accel/include/accelerator.h` with `IAccelerator` interface
2. Create `accel/include/accel_device_macros.h` with compilation-dispatch
   macros (`TUTTI_DEVICE`, `TUTTI_GLOBAL`, `TUTTI_FORCEINLINE`,
   `TUTTI_HOST_DEVICE`, `TUTTI_ATOMIC_U32`, `TUTTI_LAUNCH_KERNEL`)
3. Create `accel/cuda/` with `CudaAccelerator` wrapping all CUDA calls
4. Replace `cudaStream_t` with `AccelStream` (= `void*`) in ALL public headers:
   - `coordinator/include/coordinator.h`
   - `io_engine/include/io_engine.h`
   - `io_engine/include/backend_provider.h`
   - `memory/include/memory_subsystem.h`
5. Replace direct `cudaMalloc`/`cudaFree`/`cudaMemcpy` with
   `IAccelerator` method calls in implementation files
6. Replace `nvm_dma_map_data_device` with `IAccelerator::dma_map()` in
   `memory/` and `device_manager/`
7. Replace `__device__` / `__global__` / `__forceinline__` /
   `cuda::atomic` / `<<<>>>` with macro equivalents in all `.cuh` files

**Verification**: All existing smoke tests pass with no behavior change.

**Dependency**: None (can start immediately).

---

### Phase 2: Extract `filesystems/` from `nvme_storage/`

**Steps**:
1. Create `filesystems/include/filesystem.h` with `IFilesystem` interface
2. Move `fiemap_helper.h` from `nvme_storage/include/` to
   `filesystems/ext4_fiemap/`
3. Implement `Ext4FiemapFilesystem : IFilesystem` wrapping the FIEMAP logic
4. `nvme_storage` calls `IFilesystem::resolve()` instead of doing FIEMAP directly
5. Add `filesystems/raw_passthrough/` — no-op filesystem for raw LBA ranges

**Verification**: File creation/open/delete still works. Raw NVMe path
works (new capability).

**Dependency**: None (can proceed in parallel with Phase 1).

---

### Phase 3: Absorb `nvme_storage/` + `libnvm` into `backends/local_nvme/`

**Steps**:
1. Rename `backends/local/` to `backends/local_nvme/`
2. Rename `NVMeService/` to `nvmeservice/`
3. Absorb `nvme_storage/` into `backends/local_nvme/`:
   - `nvme_storage/include/*.h` -> `backends/local_nvme/include/`
   - `nvme_storage/src/` -> `backends/local_nvme/src/`
4. Absorb `libnvm` into `backends/local_nvme/`:
   - `backends/local/nvme/libnvm/*.h` -> `backends/local_nvme/include/libnvm/`
   - `backends/local/nvme/libnvm/*.cpp` -> `backends/local_nvme/src/libnvm/`
5. Move from `device_manager/include/` to `backends/local_nvme/include/`:
   - `local_nvme_device.h`
   - `nvme_queue_group.h`
   - `local_nvme_direct_registry.h`
   - `nvmeservice_backed_registry.h`
6. Move from `io_engine/include/local_nvme/` to `backends/local_nvme/include/`:
   - `host_batch_builder.h`
   - `launch_batch.h`
   - `local_nvme_io_engine.h`
   - `nvme_batch.h`
7. `device_manager/` keeps only `device_registry.h` and `lease_manager.h`
8. Delete `nvme_storage/` directory (fully absorbed)
9. Delete `io_engine/include/local_nvme/` subdirectory (moved to backend)

**Risk**: Large mechanical move; many include path updates.

**Verification**: All smoke tests compile and pass.

**Dependency**: Phase 2 (filesystems extracted first).

---

### Phase 4: Wire `IBackendProvider` SPI into runtime

**Steps**:
1. Implement `LocalNvmeBackend : IBackendProvider` in
   `backends/local_nvme/` — wraps current `LocalNvmeIoEngine` logic
2. Make `IIoEngine` backend-neutral (clean break):
   - Replace `NvmeBatchInputTensor` with generic
     `BufferDescriptorBatch` + `IORequestBatch`
   - `submit_batch()` dispatches to `IBackendProvider::launch_batch_gpu_stream()`
3. `Coordinator` holds `IBackendProvider*` instead of `LocalNvmeIoEngine*`
4. Backend registration: `LocalNvmeBackend::initialize()` called after
   queues + DMA contexts ready
5. Update all `examples/` and `adapters/` callers for the new API
   (clean break, no deprecation wrappers)

**Risk**: This is the deepest change — the IO submission path is
rewritten to go through the SPI. Must preserve current performance.

**Verification**: All smoke tests pass; benchmark shows no regression.

**Dependency**: Phase 3 (types in place), Phase 1 (accel abstraction).

---

### Phase 5: Introduce `StorageTarget` and make file optional

**Steps**:
1. Define `StorageTarget` in `coordinator/include/storage_target.h`
2. Add `IBackendProvider::acquire_target_handle(StorageTarget)` /
   `release_target_handle()`
3. `GpuFile` becomes one way to produce a `StorageTarget` (file-backed);
   raw NVMe targets bypass `block_storage/` entirely
4. Add `Coordinator::acquire_raw_target(StorageTarget)` for raw NVMe access
5. Update `block_storage/` to produce `StorageTarget` values

**Verification**: File path still works. Raw NVMe path works (new).

**Dependency**: Phase 4 (SPI wired).

---

### Phase 6: Consolidate kernel modules with DKMS

**Steps**:
1. Merge `snvme-5.15.0-public/` and `snvme-5.4.241-1-tlinux4-0017/`
   into a single `snvme/` source tree
2. Extract common modifications into `patches/snvme-base.patch`
3. Create version-specific patches (`snvme-5.15.patch`, `snvme-5.4.patch`)
4. Add `dkms.conf` and version-dispatching `Makefile`
5. Add kernel adaptation layer (`snvme/adapt/nvme_compat.h`)

**v0.1 target**: 5.4 (TencentOS) and 5.15 (Ubuntu) only. 6.x support
deferred to future versions.

**Risk**: Patch merging requires careful diff analysis.

**Verification**: Module builds and loads on both 5.4 and 5.15.

**Dependency**: None (can proceed in parallel).

---

### Phase 7: GDS backend (first new backend)

**Steps**:
1. Create `backends/gds/` implementing `IBackendProvider`
2. Use NVIDIA cuFile API for data path (no kernel module needed)
3. Pair with `filesystems/ext4_fiemap/` for namespace resolution
4. Validates the SPI design — adding a backend requires no changes to
   `coordinator/`, `memory/`, `io_engine/`, or `device_manager/`

**Verification**: GDS backend passes smoke tests on NVIDIA hardware.

**Dependency**: Phase 4 (SPI wired), Phase 5 (StorageTarget).

---

### Future: RDMA backend (community issue)

**Steps** (framework only; implementation by community contributors):
1. Create `backends/local_rdma/` implementing `IBackendProvider`
2. See [storage-extensibility.md](../design/storage-extensibility.md) section 4
   for the RDMA backend design
3. File a GitHub issue with the design as a starting point for contributors

**Dependency**: Phase 4 (SPI wired), Phase 5 (StorageTarget).

---

## 5. Dependency Graph

```
Phase 1 (accel abstraction + macros)  --------.
                                               |
Phase 2 (extract filesystems/)  ---------------|
                                               |
Phase 6 (DKMS kernel modules)  ----------------' (independent)
                                               |
                                               v
Phase 3 (absorb nvme_storage + libnvm, move types to backends/)
                                               |
                                               v
Phase 4 (wire IBackendProvider SPI, generic IIoEngine)
                                               |
                                               v
Phase 5 (StorageTarget, file-optional)
                                               |
                                               v
Phase 7 (GDS backend -- first new backend validation)

Future:  RDMA backend (community issue)
```

**Parallelizable**: Phases 1, 2, 6 can run in parallel.

## 6. What Stays Unchanged

- **`coordinator/` directory** — stays as the orchestrator + noun home.
  Loses `cuda_runtime.h` dependency; gains `storage_target.h`.
- **`memory/` directory** — stays, but headers lose `cuda_runtime.h`
  and `nvm_types.h` dependencies.
- **`io_engine/` directory** — stays, but `local_nvme/` subdirectory
  moves to `backends/local_nvme/`. `IIoEngine` becomes generic.
- **`block_storage/` directory** — stays, but becomes one filesystem
  option, not the only path.
- **`device_manager/` directory** — stays, thinned to registry + lease.
- **`adapters/`, `scripts/`** — unchanged.
- **`examples/`** — unchanged location; updated for new API (clean break).

## 7. Effort Estimates

| Phase | Scope | Complexity | Dependencies |
|-------|-------|------------|--------------|
| Phase 1 | accel/ abstraction + macros | Medium (new layer) | None |
| Phase 2 | Extract filesystems/ | Medium (move + rewire) | None |
| Phase 3 | Absorb nvme_storage + libnvm, move types | Medium (mechanical, wide) | Phase 2 |
| Phase 4 | Wire IBackendProvider SPI | High (rewrite IO path) | Phase 1, 3 |
| Phase 5 | StorageTarget + file-optional | Medium (new types) | Phase 4 |
| Phase 6 | DKMS kernel modules | Medium (patch merging) | None |
| Phase 7 | GDS backend | Medium (new backend) | Phase 4, 5 |

## 8. Risk Mitigation

1. **Performance regression** (Phase 4): Benchmark before/after with
   existing smoke tests. The SPI dispatch is one virtual call per batch,
   not per IO — negligible.

2. **Include path explosion** (Phase 3): Use a script to automate
   `#include` path updates. Verify with `grep` after each move.

3. **Kernel module breakage** (Phase 6): Keep old directories until the
   new DKMS build is verified on both kernel versions.

4. **CUDA compilation issues** (Phase 1 macros): Do macro replacement
   in one commit per file; compile-test after each.

## 9. Success Criteria

After restructuring:

1. **No `cuda_runtime.h` in any public header** — verifiable via `grep`.
2. **No `nvm_types.h` outside `backends/local_nvme/`** — verifiable via `grep`.
3. **`IIoEngine` takes generic types, not `NvmeBatchInputTensor`**.
4. **A new backend can be added by implementing only `IBackendProvider`** —
   no changes to `coordinator/`, `memory/`, `io_engine/`, or `device_manager/`.
5. **Raw NVMe access works without files** — `StorageTarget { NVME_RAW }`.
6. **Kernel module builds via DKMS on 5.4 and 5.15**.
7. **All existing smoke tests pass** with no behavior change.
8. **GDS backend works** as proof that the SPI supports a second backend.

## 10. Long-Term Vision and Candidate Framework Support (Exploratory)

> **Status**: Exploratory. Unlike §1-§9 (confirmed decisions), this
> section sketches a longer-horizon target and candidate framework
> integrations. Nothing here is committed — each item still requires
> its own RFC and maintainer sign-off per `CONTRIBUTING.md` ("When an
> RFC or Design Discussion Is Required") before implementation begins.

### 10.1 End-State Architecture (Beyond Phase 7)

Once Phases 1-7 land, the runtime reaches the following steady-state
shape. This is the union of what every design document already
targets, not a new decision:

```
┌───────────────────────────────────────────────────────────────┐
│  adapters/        (LMCache, Mooncake, + candidates in §10.2)   │
├───────────────────────────────────────────────────────────────┤
│  api/             (stable application-facing headers)          │
├───────────────────────────────────────────────────────────────┤
│  coordinator/       memory/         device_manager/            │
├───────────────────────────────────────────────────────────────┤
│  filesystems/     (ext4_fiemap, tutti_layout, dfs_client,       │
│                     raw_passthrough)                            │
├───────────────────────────────────────────────────────────────┤
│  io_engine/       (generic IBackendProvider SPI)                │
├───────────────────────────────────────────────────────────────┤
│  accel/           (IAccelerator: cuda, rocm, sycl, cann)        │
├───────────────────────────────────────────────────────────────┤
│  backends/        (local_nvme, gds, local_rdma, ...)            │
└───────────────────────────────────────────────────────────────┘
```

Every layer above already has a design doc or a phase in this plan
except the specific adapter list, which is intentionally open — see
§10.2.

### 10.2 Candidate Upper-Layer Framework Adapters

`adapters/` today only commits to `LMCache` and `Mooncake`
(`Roadmap.md` Phase 6). Beyond those, the following are candidates a
contributor could open an RFC for; listing them here is not a
commitment to build any of them:

| Framework | Integration point | Notes |
|---|---|---|
| vLLM | Direct KV-cache swap-out/in (analyzed in [backend-spi.md](../design/backend-spi.md) §7.3) | Primary workload already analyzed; no adapter implemented yet |
| SGLang | KV-cache offload, same shape as vLLM | Would likely reuse most of `KvCacheIoAdapter` |
| TensorRT-LLM | KV-cache paging to storage | Needs its own tensor-registration glue |
| Triton Inference Server | Model / state checkpoint spill to NVMe | Different IO pattern (large sequential, not KV blocks) |
| DeepSpeed-Inference / ZeRO-Offload | Parameter / optimizer-state offload | Batch pattern differs from KV cache (larger, less frequent) |
| Ray / Ray Serve | Object spilling to NVMe-backed storage | Would use the raw `StorageTarget` path ([storage-extensibility.md](../design/storage-extensibility.md) §5), not the KV adapter |

### 10.3 Candidate Accelerator Vendors

Tracked in detail in [gpu-abstraction.md](../design/gpu-abstraction.md);
restated here for visibility of the long-term target:

| Vendor | Framework | Status |
|---|---|---|
| NVIDIA | CUDA | v0.1 baseline (only vendor supported today) |
| AMD | ROCm | Candidate — gpu-abstraction.md Phase 4 proposes a proof-of-concept |
| Intel | oneAPI / SYCL | Candidate — different compilation model, needs its own design pass |
| Huawei | CANN (Ascend) | Candidate — needs a vendor-specific DMA-mapping equivalent (gpu-abstraction.md §2.5) |

### 10.4 Candidate Distributed Filesystem Clients

Tracked in detail in [storage-extensibility.md](../design/storage-extensibility.md)
§7.5; restated here: `3FS`, `JuiceFS`, `DAOS`, `CephFS`. Each requires
its own `filesystems/dfs_client/<name>/` implementation and is
independent work — landing one does not require landing the others.

### 10.5 Governance Note

Every item in §10.2-§10.4 requires, before any code lands:

1. A design discussion / RFC per `CONTRIBUTING.md`.
2. Explicit maintainer approval before the directory is created.
3. Its own `doc/design/` entry once approved — this section is a index
   of candidates, not a substitute for that.
