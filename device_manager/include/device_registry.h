#ifndef __TUTTI_DEVICE_MANAGER_DEVICE_REGISTRY_H__
#define __TUTTI_DEVICE_MANAGER_DEVICE_REGISTRY_H__

/**
 * device_registry.h -- the device discovery / lookup service.
 *
 * Layer: Device Manager (Roadmap.md §3 layered architecture).
 *
 * Role:
 *   - The runtime asks "which devices exist, what are their
 *     capabilities?" and gets back a list of `Device*` handles.
 *   - Backends register their devices into the registry at init time
 *     (one entry per NVMe controller, RDMA NIC, GDS adapter, ...).
 *   - The registry is the single, runtime-facing source of truth for
 *     "the device fleet", instead of each caller doing its own
 *     ad-hoc discovery.
 *
 * Layer boundary:
 *   - This is the *service* contract; the runtime depends on this
 *     interface, not on a concrete implementer. The runtime-visible
 *     `Device` value type lives in `coordinator/include/device.h`.
 *   - Implementations live in `device_manager/src/` (in-process
 *     registry) or in backend daemons (e.g. NVMeService publishes
 *     its devices through a remote-backed registry implementation).
 *
 * Lifetime:
 *   - Registry-owned `Device*` handles outlive any single submission
 *     and are stable for the life of the IDeviceRegistry. Callers
 *     MUST NOT delete them.
 *   - Adding / removing devices at runtime is allowed (hotplug,
 *     daemon restart) but not required for v0.1.
 */

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tutti {

struct Device;   // coordinator/include/device.h

/**
 * Registry of devices the runtime can submit IO against.
 *
 * v0.1 keeps the surface intentionally minimal — just enumeration
 * and lookup. Capability filtering is done by callers reading
 * `Device::capabilities`. Hot-plug / removal callbacks are deferred
 * to a later version.
 */
class IDeviceRegistry {
public:
    virtual ~IDeviceRegistry() = default;

    /// Bring the registry online: discover / attach / connect every
    /// device described by the implementation's construction config.
    /// MUST be called AFTER any fork() the application does (libnvm
    /// pulls in CUDA, which is fork-hostile).  Returns false on
    /// failure; partial state is rolled back internally before
    /// returning.  Implementations do bring-up here, never in the
    /// constructor.
    virtual bool Open() = 0;

    /// Tear the registry down: close controllers / disconnect daemon
    /// sessions.  Idempotent.  The destructor also runs this, so an
    /// explicit Close() is optional but lets callers sequence teardown
    /// deterministically.
    virtual void Close() = 0;

    /// Number of devices currently in the registry.
    virtual std::size_t device_count() const = 0;

    /// Returns the device at index `i` (0 <= i < device_count), or
    /// nullptr if `i` is out of range. Indices are stable for the
    /// life of the registry; new devices append.
    virtual const Device* device_at(std::size_t i) const = 0;

    /// Lookup by dense `device_id`. Returns nullptr if no device
    /// with that id is registered.
    virtual const Device* find_by_id(int32_t device_id) const = 0;

    /// Snapshot of every device currently in the registry.  Order
    /// matches `device_at(0..device_count())`.  Returned pointers
    /// follow the same lifetime rules as `device_at()` -- valid for
    /// the life of the registry, must not be deleted.
    virtual std::vector<const Device*> list() const = 0;
};

} // namespace tutti

#endif // __TUTTI_DEVICE_MANAGER_DEVICE_REGISTRY_H__
