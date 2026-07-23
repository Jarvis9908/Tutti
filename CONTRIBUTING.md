# Contributing to Tutti

Thank you for your interest in contributing to Tutti! This document provides guidelines and information to help you get started.

It defines the collaboration rules for human contributors and AI contributors working in this repository. Use it together with:

- [`README.md`](README.md)
- [`Roadmap.md`](Roadmap.md)

## Ways to Contribute

- **Report bugs** — open an issue with steps to reproduce (see [Reporting bugs](#reporting-bugs) for the environment info to include).
- **Request features** — describe your use case and proposed solution in an issue.
- **Improve documentation** — fix typos, clarify explanations, or add examples.
- **Submit code** — fix bugs, implement features, or improve performance (see the rules below).

## Getting Started

### Prerequisites

- Linux, kernel 5.15.x (or the 5.4.241 tlinux4 lineage) — must match an `snvme-<tag>` module baseline
- NVIDIA GPU + CUDA Toolkit ≥ 12.6 (`nvcc`)
- CMake ≥ 3.18 and a C++20 compiler
- protobuf / gRPC / uuid / yaml-cpp / libunwind — installed automatically by `scripts/prepare_env.sh`
- A **throwaway** NVMe SSD for the destructive driver tests

### Build

Full instructions live in [`doc/build_and_test.md`](doc/build_and_test.md); the short version:

```bash
# 1. One-shot dependency setup (protobuf / gRPC / uuid / ...; generates CMakePresets.json)
bash scripts/prepare_env.sh

# 2. Configure and build (presets: default / debug / release / system)
cmake --preset default
cmake --build --preset default

# 3. Install the snvme kernel module (produces build/module/snvme-core.ko + snvme.ko)
cd build && make insmod
lsmod | grep snvme          # expect snvme + snvme_core
```

Notes:

- CMake auto-matches the `snvme-<tag>` kernel-module baseline against `uname -r`; override with `-DSNVME_KERNEL_VERSION=<tag>`.
- Personal overrides (custom `VCPKG_ROOT`, CUDA path, ...) belong in `CMakeUserPresets.json`. Both preset files are gitignored — do not commit them.
- After any driver change you must reload the `.ko` (`sudo bash scripts/reset_snvme.sh`) — a stale module returns `-ENOTTY` on valid ioctls.

## Running tests

Two test suites, both covered in detail in [`doc/build_and_test.md`](doc/build_and_test.md):

**1. SNVMe driver smoke ladder** (`backends/local/kernel_modules/test/`) — safest → most destructive. Run each rung only after the one below passes:

```bash
cd backends/local/kernel_modules/test && make
export TGT=<your-test-device-BDF>     # pick a throwaway disk: scripts/pci_topology_check.sh

sudo ./snvme_smoke        $TGT        # safe: UAPI smoke, no bind
sudo ./snvme_smoke_qgroup $TGT        # safe: queue-group lifecycle
sudo ./snvme_smoke_addq    $TGT       # destructive: binds the controller
sudo ./snvme_smoke_recycle $TGT       # destructive: queue recycle via raw admin cmd
sudo ./snvme_smoke_io      $TGT       # destructive: 23-phase CPU end-to-end, WRITES LBA — the authoritative CPU-side gate
sudo ./snvme_smoke_gpu --gpu 0 --rounds 4 $TGT   # GPU hosts only: the authoritative GPU-side gate
```

🛑 The destructive rungs detach the in-tree `nvme` driver and `snvme_smoke_io` **writes to disk**. Use a throwaway controller only.

**2. Layer smoke tests** (`examples/`) — per-subsystem smoke binaries (`memory_smoke`, `registry_smoke`, `io_engine_smoke`, `nvme_storage_*_smoke`, `block_storage_*_smoke`, `e2e_smoke`, `kv_cache_adapter_smoke`, ...) built with the main presets.

For PRs: run the relevant smoke tests before opening, and state what you ran in the PR description.

## Reporting bugs

1. Gather your environment: kernel version (`uname -r`), GPU info (`nvidia-smi`), loaded modules (`lsmod | grep snvme`), and any relevant `dmesg` output.
2. Open an issue with the environment info plus clear reproduction steps. If a smoke test fails, include which rung failed (see the test ladder above) and its `[FAIL] step=<N> ... errno=<E>` line.

## Contact

Questions, adaptation proposals, or interest in becoming a community maintainer / manager — reach out to Shi Qiu at ryeqiu@tencent.com.

For general discussion or task coordination, comment on the issue.
