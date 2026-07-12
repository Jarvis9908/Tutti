/**
 * tutti_daemon.cpp -- Tutti NVMe service daemon entry (R9.6).
 *
 * The owner half of the SERVICE_CLIENT data plane.  This process is
 * the sole owner of every NVMe controller named in sys_config.yaml:
 * it does the libnvm B3 bring-up (chrdev_create + cap + bind + probe),
 * installs per-GPU view symlinks, and serves gRPC so that Coordinator
 * instances running in SERVICE_CLIENT mode (see
 * coordinator/include/coordinator_config.h) can attach as libnvm
 * clients and build their own user queue groups.
 *
 * No quota ledger is maintained here -- the kernel owns user QID
 * accounting; the daemon only brokers the chrdev/bind lease.  This is
 * a thin wrapper over nvmeservice::ServiceState (R9.q1 lean: reuse the
 * SessionBroker, wrap only the entry point), giving the daemon a
 * single tutti-branded entry that lives alongside the rest of the
 * examples/ tree.
 *
 * Pairing:
 *   1. Start this daemon on the NVMe-owning host:
 *        sudo ./tutti_daemon --config sys_config.yaml
 *   2. Point a SERVICE_CLIENT Coordinator / e2e_smoke at it:
 *        sudo ./e2e_smoke --cuda 0 --service 127.0.0.1:50051 \
 *                         --dev-id 0 --dev-id 1
 */

#include "nvmeservice_config.h"
#include "nvmeservice_server.h"
#include "nvmeservice_state.h"

#include <grpcpp/grpcpp.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

/*
 * Async-signal-safe shutdown handshake.  Calling grpc::Server::
 * Shutdown() (which touches an absl::Mutex) directly from a signal
 * handler is unsafe -- abseil's RAW_CHECK aborts on "illegal
 * recursion into Mutex code".  We just flip an atomic flag in the
 * handler (async-signal-safe) and let the main thread poll it and run
 * the real teardown.
 */
static std::atomic<int> g_stop{0};

static void on_signal(int /*sig*/) {
    g_stop.store(1, std::memory_order_relaxed);
}

static void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s --config <sys_config.yaml>\n"
        "\n"
        "Brings up every NVMe named in the config as the owner and runs\n"
        "the tutti NVMe service daemon until SIGINT/SIGTERM.\n",
        prog);
}

int main(int argc, char** argv) {
    std::string config_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--config" || arg == "-c") && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    if (config_path.empty()) {
        std::fprintf(stderr, "Missing --config\n");
        print_usage(argv[0]);
        return 1;
    }

    std::string parse_err;
    auto cfg_opt = nvmeservice::parse_config_file(config_path, &parse_err);
    if (!cfg_opt.has_value()) {
        std::fprintf(stderr, "Config parse failed: %s\n", parse_err.c_str());
        return 1;
    }
    const auto& cfg = cfg_opt.value();

    std::shared_ptr<nvmeservice::ServiceState> state;
    try {
        state = std::make_shared<nvmeservice::ServiceState>(cfg);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ServiceState init failed: %s\n", e.what());
        return 1;
    }

    state->start_reaper();

    nvmeservice::NvmeServiceImpl svc(state);

    grpc::ServerBuilder builder;
    int bound_port = 0;
    builder.AddListeningPort(cfg.grpc.endpoint,
                             grpc::InsecureServerCredentials(),
                             &bound_port);
    builder.RegisterService(&svc);

    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    if (!server) {
        std::fprintf(stderr, "Failed to start gRPC server on %s\n",
                     cfg.grpc.endpoint.c_str());
        state->stop_reaper();
        return 1;
    }

    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);
    std::signal(SIGPIPE, SIG_IGN);

    std::cout << "tutti_daemon listening on " << cfg.grpc.endpoint
              << " (port " << bound_port << ")\n";
    std::cout << "Owned devices:\n";
    for (const auto& d : state->list_devices()) {
        std::cout << "  device_id=" << d.device_id
                  << " pci="  << d.pci_addr
                  << " snvme=" << d.snvme_dev_path
                  << " ns="   << d.namespace_id
                  << " max_user_qid=" << d.max_user_qid
                  << "\n";
    }
    std::cout.flush();

    /* Poll the signal flag from the main thread (NOT the handler). */
    while (g_stop.load(std::memory_order_relaxed) == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "Shutting down...\n";
    std::cout.flush();
    server->Shutdown();
    server->Wait();
    state->stop_reaper();
    std::cout << "tutti_daemon exited cleanly.\n";

    return 0;
}
