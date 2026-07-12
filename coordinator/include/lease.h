#ifndef __TUTTI_RUNTIME_LEASE_H__
#define __TUTTI_RUNTIME_LEASE_H__

/**
 * lease.h -- the runtime-visible Lease handle (noun).
 *
 * Layer: Core Runtime (Roadmap.md §3 object model).
 *
 * Role:
 *   - A Lease is a generic "you may use this resource until it expires"
 *     token. The first concrete user is the NVMeService client model
 *     (queue ranges leased per client; reaper reclaims dead ones), but
 *     the abstraction also covers future leases:
 *        * RDMA QP lease (multi-process NIC sharing)
 *        * Mount-point lease (filesystem ownership in shared backends)
 *        * Bandwidth quota lease (QoS scheduling)
 *
 * Layer split:
 *   - This header defines the value type (`Lease`, `LeaseKind`) that
 *     callers receive and pass around. It is intentionally free of
 *     device_manager dependencies.
 *   - The lifecycle service that issues, refreshes, and reaps leases
 *     lives in `device_manager/include/lease_manager.h` as the
 *     `ILeaseManager` interface. The split keeps the runtime-visible
 *     noun (this file) decoupled from the cross-process management
 *     plane (NVMeService daemon, future RDMA-pool daemons, etc).
 *
 * Lifetime:
 *   - Lease handles are owned by the requester. The lease manager
 *     tracks them by lease_id. Releasing a Lease early via
 *     ILeaseManager::release_lease() is fine; otherwise the heartbeat
 *     keeps it alive.
 *   - The "what was leased" payload is opaque (`payload`); each
 *     resource type defines its own struct stored at that pointer,
 *     owned by the issuing ILeaseManager for the life of the lease.
 */

#include <cstdint>
#include <chrono>
#include <string>

namespace tutti {

/**
 * Tag identifying which resource shape `payload` carries. New kinds
 * append; numeric values stable.
 */
enum class LeaseKind : uint32_t {
    NVME_QUEUE_RANGE = 0,   // payload = pointer to a nvme-queue-range descriptor
    RDMA_QP          = 1,   // future
    BANDWIDTH_QUOTA  = 2,   // future
};

/**
 * One outstanding lease. Heartbeat-managed.
 *
 * `last_heartbeat` is updated by ILeaseManager::heartbeat() and
 * inspected by the reaper. `client_pid` + `client_pid_starttime` form
 * the (PID, /proc/<pid>/stat starttime) tuple the reaper uses to
 * defeat PID reuse, mirroring the NVMeService design.
 */
struct Lease {
    std::string lease_id;          // opaque, server-issued (typically a hex token)
    LeaseKind   kind;
    int32_t     device_id;         // Device this lease draws from
    uint32_t    client_pid;        // 0 if lease is in-process / no PID identity
    uint64_t    client_pid_starttime;  // /proc/<pid>/stat field 22; 0 if N/A
    std::chrono::steady_clock::time_point last_heartbeat;

    // Lease-kind-specific payload. ILeaseManager-owned for the life
    // of the lease. Cast based on `kind`.
    void* payload;

    // Lease parameters chosen at issue time.
    uint32_t heartbeat_interval_sec;
    uint32_t timeout_sec;
};

// NOTE:
//   The `ILeaseManager` lifecycle interface (heartbeat / release / ...)
//   has moved to `device_manager/include/lease_manager.h`. Code that
//   only needs to hold a `Lease` value should keep including this
//   header; code that drives the lease lifecycle should include the
//   device_manager header instead.

} // namespace tutti

#endif // __TUTTI_RUNTIME_LEASE_H__
