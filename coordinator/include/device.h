#ifndef __TUTTI_RUNTIME_DEVICE_H__
#define __TUTTI_RUNTIME_DEVICE_H__

/**
 * device.h -- runtime-level handle for one backend-managed device.
 *
 * Layer: Core Runtime (Roadmap.md §3 object model).
 *
 * Role:
 *   - A `Device` is the runtime-visible identity of one storage
 *     resource: an NVMe controller, an RDMA NIC + queue pair pool,
 *     a GDS adapter, etc.  Upper layers refer to devices via
 *     `device_id` and a `Device*` handle they obtain from the Runtime.
 *   - The Device is intentionally thin: identity + capabilities +
 *     a backend-private opaque handle.  Everything else (queue
 *     lifecycle, IO submission, mount paths, filesystems) lives in
 *     adjacent runtime types or hangs off `backend_private`.
 *
 * What's deliberately NOT here:
 *   - GPU bindings (those are Lease-time decisions; multiple GPUs
 *     may share one Device).
 *   - Filesystem mount paths (StorageTarget territory in R9+).
 *   - Per-queue scheduling state (the device_manager's queue group
 *     owns this for v0.1).
 *
 * Lifetime:
 *   - The Runtime owns Device instances. Holders carry `Device*` and
 *     MUST NOT delete them. A Device's identity is stable for the
 *     life of the Runtime.
 */

#include <cstdint>
#include <string>

#include "capability_set.h"

// io_engine forward decls
#include "../../io_engine/include/backend_type.h"

namespace tutti {

/**
 * Runtime-level device descriptor. Field set is intentionally small;
 * extending requires bumping a version field on the Runtime.
 */
struct Device {
    // ---- Identity ----------------------------------------------------
    int32_t      device_id;        // dense index inside the Runtime
    BackendType  backend_type;     // which backend manages this device
    std::string  pci_addr;         // bus identifier (PCI for NVMe, IB GUID for RDMA, ...)
    std::string  display_name;     // human-readable, e.g. "Samsung 980 Pro @ 0000:50:00.0"

    // ---- Capabilities -----------------------------------------------
    CapabilitySet capabilities;

    // ---- Backend-private opaque payload -----------------------------
    // Backends may stash a private handle (e.g. a `Controller*` for
    // local_nvme, an `ibv_context*` for RDMA) so they can navigate
    // back from `Device` to their concrete state without a side table.
    // Upper layers MUST treat this as opaque.
    void* backend_private;
};

} // namespace tutti

#endif // __TUTTI_RUNTIME_DEVICE_H__
