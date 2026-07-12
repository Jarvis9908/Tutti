#ifndef __TUTTI_EXAMPLES_COMMON_REGISTRY_CLI_H__
#define __TUTTI_EXAMPLES_COMMON_REGISTRY_CLI_H__

/**
 * registry_cli.h -- shared CLI surface + registry bring-up for
 * layer-specific smokes that talk directly to nvme_storage /
 * block_storage / io_engine (below Coordinator).
 *
 * Every smoke below Coordinator used to hard-code IN_PROCESS-only
 * bring-up (LocalNvmeDirectRegistry).  This means none of them could
 * be pointed at a running tutti_daemon (SERVICE_CLIENT) the way
 * e2e_smoke / kv_cache_adapter_smoke already can -- which matters
 * for "repeat client-mode runs against one long-lived daemon,
 * exercising the daemon's own lease/fault-tolerance path" testing.
 *
 * This header gives every such smoke the SAME `--service <endpoint>
 * --dev-id N --dev-id M` convention e2e_smoke/kv_cache_adapter_smoke
 * already use (deliberately NOT the older `--mode=direct|service
 * --device=csv` convention registry_smoke/nvme_storage_gpu_smoke
 * happen to use -- those two already work; this is for the smokes
 * that had NO service-mode support at all).
 *
 * Usage (typical smoke main()):
 * @code
 *   tutti_examples::RegistryCliOptions opt;
 *   for (int i = 1; i < argc; ++i) {
 *       if (tutti_examples::parse_registry_cli_arg(opt, argc, argv, i)) continue;
 *       // ... smoke's own flags (--cap, positional BDFs, etc) ...
 *   }
 *   auto reg = tutti_examples::open_registry(opt, true);   // build_queue_group
 *   if (reg.ptr == nullptr) STEP_FAIL("registry Open()");
 *   // use reg.ptr->device_count() / device_at(i) as before -- mode-agnostic.
 * @endcode
 *
 * Layer boundary: header-only, examples-only.  Pulls in
 * device_manager's two registry headers (already CUDA-free at the
 * type level) -- no new library dependency for callers that already
 * link tutti_device_manager (every smoke here does).
 */

#include "device_registry.h"
#include "local_nvme_direct_registry.h"
#include "nvmeservice_backed_registry.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace tutti_examples {

/// Parsed CLI options shared by every direct/service smoke.  Callers
/// own the struct and keep parsing their own additional flags for
/// anything not covered here (positional PCI BDFs, --cap, --cuda,
/// etc. -- see parse_registry_cli_arg's doc for which of those THIS
/// header already recognizes).
struct RegistryCliOptions {
    bool                     service_mode     = false;
    std::string              service_endpoint = "127.0.0.1:50051";
    std::vector<int32_t>      dev_ids;      // SERVICE_CLIENT: daemon-local device ids
    std::vector<std::string>  pci_addrs;    // IN_PROCESS: PCI BDFs (positional args)
    int32_t                  cuda_dev         = 0;
    uint32_t                 kernel_ioq_cap   = 32;  // IN_PROCESS only
    uint32_t                 num_user_queues  = 4;   // per device, both modes
};

/// Tries to consume argv[i] (and, for flags taking a value, argv[i+1]
/// too -- advancing `i` past it) as one of the flags this header
/// understands: --service, --dev-id (repeatable), --cuda, --cap,
/// --queues.  Returns true iff it recognized and consumed the arg;
/// callers should `continue` their own parsing loop on a true return.
/// A bare (non-flag) positional token is NOT consumed here -- callers
/// collect those into `opt.pci_addrs` themselves so this header
/// doesn't have to guess which positional-arg convention a given
/// smoke wants.
inline bool parse_registry_cli_arg(RegistryCliOptions& opt, int argc, char** argv, int& i) {
    const char* a = argv[i];
    if (!std::strcmp(a, "--service") && i + 1 < argc) {
        opt.service_mode     = true;
        opt.service_endpoint = argv[++i];
        return true;
    }
    if (!std::strcmp(a, "--dev-id") && i + 1 < argc) {
        opt.dev_ids.push_back((int32_t)std::atoi(argv[++i]));
        return true;
    }
    if (!std::strcmp(a, "--cuda") && i + 1 < argc) {
        opt.cuda_dev = std::atoi(argv[++i]);
        return true;
    }
    if (!std::strcmp(a, "--cap") && i + 1 < argc) {
        opt.kernel_ioq_cap = (uint32_t)std::atoi(argv[++i]);
        return true;
    }
    if (!std::strcmp(a, "--queues") && i + 1 < argc) {
        opt.num_user_queues = (uint32_t)std::atoi(argv[++i]);
        return true;
    }
    return false;
}

/// Validates the mode-specific required fields are present, printing
/// a clear diagnostic (mirroring every existing smoke's own
/// "usage()+return 1" convention) if not.  Callers still enforce
/// their OWN shard-count expectations (e.g. "needs exactly 2") on top
/// of this -- this only checks "is there enough to attempt Open() at
/// all".
inline bool validate_registry_cli(const RegistryCliOptions& opt, const char* prog) {
    if (opt.service_mode) {
        if (opt.dev_ids.empty()) {
            std::fprintf(stderr,
                "%s: --service mode needs at least one --dev-id\n", prog);
            return false;
        }
    } else if (opt.pci_addrs.empty()) {
        std::fprintf(stderr,
            "%s: IN_PROCESS mode needs at least one PCI BDF\n", prog);
        return false;
    }
    return true;
}

/// Owns whichever concrete registry got opened; exposes it via the
/// IDeviceRegistry* base so callers stay mode-agnostic past this
/// point -- mirrors every layer's own "consume Device*, don't care
/// which registry produced it" convention.  `ptr == nullptr` means
/// Open() failed (or `opt` was never validated); callers STEP_FAIL.
struct OpenedRegistry {
    std::unique_ptr<tutti::LocalNvmeDirectRegistry>   direct;
    std::unique_ptr<tutti::NvmeServiceBackedRegistry> service;
    tutti::IDeviceRegistry*                           ptr = nullptr;
};

/// Builds + Opens the right registry from `opt`.  `build_queue_group`
/// mirrors LocalNvmeDirectConfig/NvmeServiceBackedRequest's own knob
/// (true for smokes that submit GPU-direct NVMe IO; false for
/// host-only ones like the plain nvme_storage_smoke).
inline OpenedRegistry open_registry(const RegistryCliOptions& opt, bool build_queue_group) {
    OpenedRegistry out;
    if (opt.service_mode) {
        std::vector<tutti::NvmeServiceBackedRequest> reqs;
        reqs.reserve(opt.dev_ids.size());
        for (int32_t did : opt.dev_ids) {
            tutti::NvmeServiceBackedRequest r{};
            r.daemon_device_id  = did;
            r.cuda_device       = opt.cuda_dev;
            r.num_queues        = (int32_t)opt.num_user_queues;
            r.build_queue_group = build_queue_group;
            r.num_user_queues   = opt.num_user_queues;
            reqs.push_back(std::move(r));
        }
        out.service = std::make_unique<tutti::NvmeServiceBackedRegistry>(
            opt.service_endpoint, std::move(reqs));
        if (out.service->Open()) out.ptr = out.service.get();
    } else {
        std::vector<tutti::LocalNvmeDirectConfig> cfgs;
        cfgs.reserve(opt.pci_addrs.size());
        for (const auto& bdf : opt.pci_addrs) {
            tutti::LocalNvmeDirectConfig c{};
            c.pci_addr          = bdf;
            c.kernel_ioq_cap    = opt.kernel_ioq_cap;
            c.build_queue_group = build_queue_group;
            c.cuda_device       = opt.cuda_dev;
            c.num_user_queues   = opt.num_user_queues;
            cfgs.push_back(std::move(c));
        }
        out.direct = std::make_unique<tutti::LocalNvmeDirectRegistry>(std::move(cfgs));
        if (out.direct->Open()) out.ptr = out.direct.get();
    }
    return out;
}

} // namespace tutti_examples

#endif // __TUTTI_EXAMPLES_COMMON_REGISTRY_CLI_H__
