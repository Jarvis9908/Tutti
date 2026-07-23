# Tutti Roadmap

> The previous v0.1 architecture baseline is archived at
> [`doc/history/roadmap-v0.1.md`](doc/history/roadmap-v0.1.md).

## 1. Kernels

- Extend the `snvme` kernel module to more Linux kernel versions.
- Optimizations for NVMe devices.

## 2. Accelerator cards (P2P backends)

Tutti is currently designed for NVIDIA GPUs (H100 / H20 / A100 / L40s).
The plan: NVIDIA ✅ → AMD (ROCm), Huawei Ascend NPU, MetaX, and others.

## 3. AI applications & scenarios

- vLLM PoC completed — a high-performance KV pool that can be extended
  to GNN, RAG, and more scenarios.
- Future plan: upstream into communities such as LMCache and vLLM.

## 4. Scale OUT / up

- Explore DPU and NVMe-oF to scale GPU-initiated I/O out across nodes.
- Explore NVLink and Huawei UB to scale GPU-initiated I/O up within a node.
