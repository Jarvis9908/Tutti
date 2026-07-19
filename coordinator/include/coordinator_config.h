#ifndef __TUTTI_COORDINATOR_COORDINATOR_CONFIG_H__
#define __TUTTI_COORDINATOR_COORDINATOR_CONFIG_H__

/**
 * coordinator_config.h -- bring-up configuration for the Coordinator.
 *
 * Layer: Coordinator (top of the layer cake).
 *
 * The Coordinator owns the whole local-NVMe data plane and brings it
 * up in dependency order from a single config:
 *
 *   <device registry>          (device_manager)
 *     -> HostFsBackedNvmeStorage   (nvme_storage)
 *       -> HostFsBackedBlockStorage  (block_storage)
 *         -> HostDeviceMemorySubsystem (memory)
 *           -> LocalNvmeIoEngine        (io_engine)
 *
 * Two bring-up modes select which IDeviceRegistry sits at the top of
 * the chain:
 *
 *   IN_PROCESS      LocalNvmeDirectRegistry -- this process owns the
 *                   NVMe controllers (chrdev_create + bind + probe via
 *                   libnvm B3).  Single-tenant / smoke / micro-bench.
 *
 *   SERVICE_CLIENT  NvmeServiceBackedRegistry -- a running
 *                   nvmeservice_daemon owns the chrdev/bind; this
 *                   process attaches as a client (nvm_ctrl_attach_client)
 *                   and builds its own user queue group.  The daemon
 *                   only brokers the lease; no GPU memory or queue
 *                   handles cross the process boundary.
 *
 * Everything below the registry (nvme_storage .. io_engine) is
 * mode-agnostic: it consumes `const Device*` handles and never cares
 * whether they came from a direct-owned or service-attached
 * controller.
 */

#include <cstdint>
#include <string>
#include <vector>

#include "../../memory/include/memory_subsystem.h"   // DescriptorFormat

namespace tutti {

/// Which IDeviceRegistry the Coordinator builds at bootstrap.
enum class CoordinatorMode : uint32_t {
    IN_PROCESS     = 0,   ///< direct-owned NVMe (LocalNvmeDirectRegistry)
    SERVICE_CLIENT = 1,   ///< attach via nvmeservice_daemon (NvmeServiceBackedRegistry)
};

struct CoordinatorConfig {
    /// Bring-up mode -- selects the registry implementation.  The
    /// rest of the fields split into mode-specific and shared groups
    /// (see comments).  Defaults to the historical IN_PROCESS path.
    CoordinatorMode mode = CoordinatorMode::IN_PROCESS;

    // ---- IN_PROCESS only --------------------------------------------

    /// NVMe controllers to bring up, by PCI BDF (e.g. "0000:4b:00.0").
    /// Each becomes one Device / NVMe-file shard slot.  Order is
    /// significant: it fixes the device_id assignment and the shard
    /// placement order GpuFileSpec refers to.  Ignored in
    /// SERVICE_CLIENT mode.
    std::vector<std::string> pci_addrs;

    /// NVM_SET_KERNEL_IOQ_CAP value per controller: how many queue
    /// pairs to reserve for the kernel blk-mq path; the rest go to
    /// the user QID pool.  0 = kernel default.  The owner sets this,
    /// so it is ignored in SERVICE_CLIENT mode (the daemon already
    /// fixed the cap when it brought the controller up).
    uint32_t kernel_ioq_cap = 32;

    // ---- SERVICE_CLIENT only ----------------------------------------

    /// gRPC endpoint of the nvmeservice_daemon, e.g.
    /// "127.0.0.1:50051" or "unix:/run/tutti/nvmeservice.sock".
    /// Required in SERVICE_CLIENT mode; ignored otherwise.
    std::string daemon_endpoint;

    /// Daemon-local device ids to open (from the daemon's
    /// ListDevices).  Order fixes the device_id / shard placement
    /// order just like pci_addrs does for IN_PROCESS.  Required in
    /// SERVICE_CLIENT mode; ignored otherwise.
    std::vector<int32_t> daemon_device_ids;

    // ---- shared (both modes) ----------------------------------------

    /// Primary CUDA device the GPU-side data plane (queue rings,
    /// PRP-list buffers, batch kernels) runs on.
    int cuda_device = 0;

    /// User-queue pairs to build per controller for on-GPU submit.
    /// In SERVICE_CLIENT mode this also bounds the lease quota the
    /// client requests from the daemon.  Bounded by the kernel's
    /// user QID pool + max_queues_per_group.
    uint32_t num_user_queues_per_device = 4;

    /// SQ/CQ ring depth (power of 2).  0 = controller max q_depth.
    uint32_t queue_depth = 0;

    /// NVMe namespace id for user-queue commands (1 on single-ns drives).
    uint32_t namespace_id = 1;

    /// Cluster-wide descriptor format, frozen at bootstrap.
    /// PRP today; SGL when every attached controller supports it.
    DescriptorFormat descriptor_format = DescriptorFormat::PRP;

    /// IO-engine NvmeBatchEntry scratch capacity.  Caps the
    /// flattened (tensor x sub-slice) entry count per submit_batch.
    uint32_t max_entries_per_batch = 256;

    /// Upper bound on how many GpuFile device handles may be
    /// concurrently GPU-resident (the L1 tier of nvme_storage's
    /// two-tier NvmeFileDeviceHandle cache; also sizes block_storage's
    /// own single-tier ShardPtrSlot pool).  A GpuFile's persistent
    /// directory entry is cheap (millions are fine), but each
    /// *GPU-resident* handle costs real GPU memory, so only the
    /// working set is kept there; the least-recently used handle is
    /// downgraded to L2 (content preserved, NOT deleted) when this
    /// cap is exceeded.
    ///
    /// NOT the same knob as `max_entries_per_batch` (that caps IO
    /// entries per kernel submission) -- this one caps concurrently
    /// *GPU-resident files*, a completely independent axis.
    ///
    /// 0 (default) = AUTO: compute from `handle_l1_gpu_budget_bytes`
    /// at bootstrap time using
    ///   per_file_bytes = kGpuFileMaxShards *
    ///       (sizeof(NvmeFileDeviceHandle) + sizeof(void*))
    ///   l1_capacity = handle_l1_gpu_budget_bytes / per_file_bytes
    /// (kGpuFileMaxShards, not the actual per-file shard count, is
    /// used as the conservative multiplier since shard count varies
    /// per GpuFile and isn't known at bootstrap time).  bootstrap()
    /// logs the computed value.
    ///
    /// Set explicitly to override the auto-compute (e.g. to force a
    /// small cap for testing eviction, as the kv_cache_adapter_smoke
    /// does).
    ///
    /// Invariant: a single submit/handle_for_batch call's distinct-
    /// file working set MUST fit under this cap for the DURATION of
    /// that call (every file it resolves must stay GPU-resident until
    /// the caller's IO kernel has run).  Adapters/TieredHandleCache
    /// check this and fail loudly, mirroring max_entries_per_batch.
    uint32_t handle_l1_capacity = 0;

    /// GPU-memory budget (bytes) the AUTO `handle_l1_capacity` sizing
    /// is allowed to spend.  Ignored when handle_l1_capacity != 0.
    /// Default 512 MiB -- a GPU also has to hold model weights and
    /// the KV-cache tensors themselves, so the GPU-resident handle
    /// tier's budget is deliberately small; the CPU-pinned L2 tier
    /// (handle_l2_host_budget_bytes below) absorbs the "millions of
    /// files, small working set" mismatch instead.
    uint64_t handle_l1_gpu_budget_bytes = 512ull * 1024 * 1024;

    /// C-1: PRP-list buffer pool budgets.  The PRP-list page (one per
    /// tensor, 4 KiB) is GPU-resident + DMA-mapped so the NVMe
    /// controller can fetch it via PRP2.  Without a pool, each tensor
    /// gets an independent cudaMalloc + nvm_dma_map_data_device with
    /// 64 KiB alignment waste (128 KiB allocated for 4 KiB of content).
    /// The pool does ONE allocation + ONE DMA map for the entire L1
    /// tier, and ONE cudaMallocHost for the L2 staging tier.
    ///
    /// L1 (GPU-resident, DMA-mapped): slots handed out at
    /// register_tensor time, released at unregister.  Default 64 MiB
    /// → 16384 slots at 4 KiB/slot.
    /// L2 (host-pinned staging): used to build PRP page content with
    /// pinned memory (faster cudaMemcpy) instead of a temporary
    /// std::vector.  Default 1 GiB → 262144 slots.
    uint64_t prp_l1_gpu_budget_bytes  = 64ull * 1024 * 1024;
    uint64_t prp_l2_host_budget_bytes = 1ull * 1024 * 1024 * 1024;

    /// Upper bound on how many GpuFile handle TEMPLATES may be
    /// CPU-pinned-resident at once (the L2 tier -- see
    /// tiered_handle_cache.h).  Every file whose handle has been
    /// built at least once lives here until genuinely evicted (rare,
    /// given the large default budget); L1 promotions/downgrades
    /// against this tier are plain memcpy, never re-walk FIEMAP.
    /// 0 (default) = AUTO from `handle_l2_host_budget_bytes`, same
    /// per_file_bytes formula as handle_l1_capacity.  MUST be >=
    /// handle_l1_capacity (L1 is backed by L2).
    uint32_t handle_l2_capacity = 0;

    /// CPU-pinned-memory budget (bytes) the AUTO `handle_l2_capacity`
    /// sizing is allowed to spend.  Ignored when handle_l2_capacity
    /// != 0.  Default 2 GiB -- comfortably covers multi-ten-million-
    /// file deployments (2 GiB / ~192 B per NvmeFileDeviceHandle ~=
    /// 11M files) while staying a modest, explicit host-RAM cost.
    uint64_t handle_l2_host_budget_bytes = 2ull * 1024 * 1024 * 1024;
};

} // namespace tutti

#endif // __TUTTI_COORDINATOR_COORDINATOR_CONFIG_H__
