# Key Designs — Performance & Low Overhead

> The five designs that keep Tutti's data path fast and its CPU overhead
> near zero, as implemented in the current codebase — each with pointers
> to the code that realizes it. For the layer-by-layer architecture see
> [system-architecture.md](system-architecture.md).

The GPU-centric data path: the CPU appears only **O(1) times per batch,
not per I/O**.

## 1. GPU io_uring — one launch, thousands of I/Os, zero CPU round-trips

A flat batch-entry array (`NvmeBatchEntry[]` — the ioctx-style submission
buffer staged in GPU memory) plays the role of io_uring's SQ: the host
fills it with ONE `cudaMemcpyAsync` and launches ONE kernel; each GPU
thread then executes a whole I/O on its own: stripe selection
(`gpu_blk % num_shards`), virtual→physical LBA resolution over the
file's extent list, queue pick (lock-free hash; SQ slots serialized by
an atomic CAS on the tail), SQE write + doorbell ring through the
GPU-mapped BAR, and completion busy-poll on the CQ phase bit. No
syscalls, no interrupts, no CPU involvement on the data path.

Code: `io_engine/src/local_nvme/nvme_batch_xfer_kernel.cu`,
`io_engine/src/local_nvme/local_nvme_io_engine.cpp`,
`nvme_storage/include/nvme_storage_device.cuh`,
`nvme_storage/include/queue_acquire_helper.cuh`

## 2. Register-time precomputation — the hot path is table lookup, not arithmetic

`register_tensor()` DMA-maps a buffer **exactly once** — PCI bus
addresses are controller-agnostic under IOMMU=pt, so one mapping serves
every bound NVMe controller — pre-splits the tensor into IO slices, and
pre-builds every PRP1/PRP2 descriptor into a contiguous GPU-resident
blob. Submit time is an O(log N) slice lookup, never PRP math.

Code: `memory/include/host_device_memory_subsystem.h` (`IoSliceTable`),
`memory/src/host_device_memory_subsystem.cu`

## 3. Two-tier caches sized for LMCache-scale file counts

A file's on-GPU handle (`NvmeFileDeviceHandle`) is **~200 bytes** — 8
inline extents plus a rare overflow buffer, mirroring NVMe's own
PRP1/PRP2 + PRP-list pattern — an order of magnitude smaller than a
naive all-inline design. Handles live in a two-tier cache: a small
GPU-resident L1 (default 512 MiB working set) backed by a large
pinned-host L2 (default 2 GiB ≈ 11M files). FIEMAP is walked exactly
once per file, at L2 admission; L1 eviction is a *downgrade*, not a
delete, so a re-touch costs one memcpy instead of a rebuild. PRP-list
pages get the same two-tier treatment — paged L2→L1 per batch —
eliminating both the 64 KiB per-tensor alignment waste and the "every
PRP page pinned in GPU memory forever" cost.

Code: `nvme_storage/include/nvme_file_device_handle.h`,
`memory/include/tiered_handle_cache.h`,
`memory/include/prp_page_cache.h`

## 4. A batched, event-frugal control plane

Cold misses in one batch are built host-side once and uploaded as ONE
contiguous `cudaMemcpyAsync` — not thousands of small copies. Slot reuse
in the GPU pool is fenced by ONE shared `cudaEvent` per producer stream
rather than one event per slot (the pool holds hundreds of thousands of
slots). All user SQ/CQ pairs are created with a single batched ioctl.
Release is CPU-side immediate, so LRU evict-then-refill never needs a
GPU sync.

Code: `memory/include/gpu_slot_pool.h`,
`device_manager/include/nvme_queue_group.h`

## 5. Multi-device striping

A `GpuFile` interleaves data across up to 4 NVMe devices in
`tensor_size` units. The shard pick happens on-GPU inside the same
kernel, so one launch fans a batch out across the whole device fleet.

Code: `block_storage/include/gpu_file_resolve.h`,
`io_engine/src/local_nvme/nvme_batch_xfer_kernel.cu`
