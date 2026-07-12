#ifndef __TUTTI_ADAPTERS_KV_CACHE_IO_ADAPTER_H__
#define __TUTTI_ADAPTERS_KV_CACHE_IO_ADAPTER_H__

/**
 * kv_cache_io_adapter.h -- KV-cache semantics adapter on top of the
 * generic Tutti io_engine (closes legacy-vs-tutti gaps D1 + D2).
 *
 * Layer: adapters/ (framework-facing; NOT part of the core runtime --
 * see Roadmap.md "Adapter Layer").  The core io_engine deliberately
 * only knows generic `(tensor, file_handle, file_byte_offset)` tuples;
 * it has no notion of K/V caches or transformer layers.  Historically
 * (GeminiFS `geminifs_batched_xfer`) that knowledge was baked INTO the
 * engine.  This adapter is the new home for it:
 *
 *   D1 (KV/layer offset semantics): reconstructs the legacy per-layer
 *       byte-offset formula
 *           standard : K @ layer * tensor_size * 2
 *                      V @ layer * tensor_size * 2 + tensor_size
 *           MLA      :   @ layer * tensor_size
 *       and turns a per-layer (K[],V[],file[]) batch into the generic
 *       NvmeBatchInputTensor list the io_engine consumes.
 *
 *   D2 (auto-chunking): the v0.1 LocalNvmeIoEngine owns ONE fixed
 *       scratch buffer (max_entries_per_batch) and rejects any batch
 *       that overflows it -- unlike legacy's 512-entry BatchIoPool
 *       loop.  This adapter greedily packs inputs into
 *       <= max_entries chunks at tensor boundaries and submits each,
 *       so callers can hand it an arbitrarily large block list.
 *
 * What this adapter does NOT do (kept out of scope on purpose):
 *   - It does not own memory registration; callers register their K/V
 *     tensors with the Coordinator (granularity == tensor_size) and
 *     pass the resulting MemoryRegion* in.  One MemoryRegion per
 *     (block, layer, K-or-V) tensor, matching legacy's one-tensor-per
 *     -registered-buffer model.  (A future optimisation can register
 *     the whole per-layer cache once and address single slices via
 *     IMemorySubsystem::lookup_io_slice; that needs a slice-scoped
 *     builder path in io_engine and is tracked separately.)
 *   - It does not manage GpuFile lifecycle / the KV index; that's the
 *     LMCache-shaped backend above this.
 *
 * Concurrency: submit_* are blocking (each chunk hard-syncs its
 * stream, per IIoEngine::submit_batch).  For per-layer stream overlap,
 * give each layer its own stream and/or its own adapter instance --
 * the underlying single shared engine scratch must not be raced (see
 * io_engine.h).
 */

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

namespace tutti {

class Coordinator;
struct MemoryRegion;
struct GpuFileHandle;

using GpuFileId = uint32_t;   // mirror of block_storage.h alias

class KvCacheIoAdapter {
public:
    /// @param coord        a booted Coordinator (owns memory + io_engine).
    /// @param tensor_size  the per-(block,layer) K (or V) tensor size in
    ///                     bytes == GpuFileSpec.tensor_shape[2] (legacy
    ///                     "len" / kv_size).  Used for the layer offset
    ///                     arithmetic; MUST match how the GpuFiles were
    ///                     laid out.
    /// @param use_mla      false: standard K/V split (kv_dim == 2).
    ///                     true : unified single-tensor layout (kv_dim == 1).
    KvCacheIoAdapter(Coordinator& coord,
                     uint64_t     tensor_size,
                     bool         use_mla = false);

    // ------------------------------------------------------------------
    // Standard K/V split (legacy geminifs_batched_read/write analogue).
    //
    // One element per block: (k_regions[i], v_regions[i]) are written to
    // / read from block i at this layer's K and V byte offsets.  All
    // vectors MUST be the same length and non-empty.
    //
    // Two file-addressing flavours:
    //   - GpuFileId  (RECOMMENDED): the runtime resolves + caches the
    //     device handle transparently via Coordinator::handle_for().
    //     Callers hold only stable, persistent ids.
    //   - GpuFileHandle*: the low-level path for callers that already
    //     acquired handles themselves.
    // ------------------------------------------------------------------

    bool batched_write(int                                 layer_idx,
                       const std::vector<MemoryRegion*>&   k_regions,
                       const std::vector<MemoryRegion*>&   v_regions,
                       const std::vector<GpuFileId>&       file_ids,
                       cudaStream_t                        stream);

    bool batched_read (int                                 layer_idx,
                       const std::vector<MemoryRegion*>&   k_regions,
                       const std::vector<MemoryRegion*>&   v_regions,
                       const std::vector<GpuFileId>&       file_ids,
                       cudaStream_t                        stream);

    bool batched_write(int                                 layer_idx,
                       const std::vector<MemoryRegion*>&   k_regions,
                       const std::vector<MemoryRegion*>&   v_regions,
                       const std::vector<GpuFileHandle*>&  file_handles,
                       cudaStream_t                        stream);

    bool batched_read (int                                 layer_idx,
                       const std::vector<MemoryRegion*>&   k_regions,
                       const std::vector<MemoryRegion*>&   v_regions,
                       const std::vector<GpuFileHandle*>&  file_handles,
                       cudaStream_t                        stream);

    // ------------------------------------------------------------------
    // Unified / MLA (single tensor per block per layer, no K/V split).
    // ------------------------------------------------------------------

    bool batched_write_unified(int                                 layer_idx,
                               const std::vector<MemoryRegion*>&   regions,
                               const std::vector<GpuFileId>&       file_ids,
                               cudaStream_t                        stream);

    bool batched_read_unified (int                                 layer_idx,
                               const std::vector<MemoryRegion*>&   regions,
                               const std::vector<GpuFileId>&       file_ids,
                               cudaStream_t                        stream);

    bool batched_write_unified(int                                 layer_idx,
                               const std::vector<MemoryRegion*>&   regions,
                               const std::vector<GpuFileHandle*>&  file_handles,
                               cudaStream_t                        stream);

    bool batched_read_unified (int                                 layer_idx,
                               const std::vector<MemoryRegion*>&   regions,
                               const std::vector<GpuFileHandle*>&  file_handles,
                               cudaStream_t                        stream);

    uint64_t tensor_size() const { return tensor_size_; }
    bool     use_mla()     const { return use_mla_; }

private:
    // Resolve GpuFileId[] -> GpuFileHandle*[] through the Coordinator's
    // transparent handle cache.  R11: async -- queues H2D fills on
    // `stream`; returned pointers are only valid for GPU work queued
    // on the SAME `stream` after this call.  Fails (false) if the
    // batch's distinct working set exceeds the cache cap, or any id
    // is unresolvable.
    bool resolve_handles(const std::vector<GpuFileId>&       file_ids,
                        std::vector<GpuFileHandle*>&        out_handles,
                        cudaStream_t                         stream);

    bool xfer(int                                 layer_idx,
              const std::vector<MemoryRegion*>&   k_regions,
              const std::vector<MemoryRegion*>&   v_regions,
              const std::vector<GpuFileHandle*>&  file_handles,
              bool                                is_read,
              cudaStream_t                        stream);

    bool xfer_unified(int                                 layer_idx,
                      const std::vector<MemoryRegion*>&   regions,
                      const std::vector<GpuFileHandle*>&  file_handles,
                      bool                                is_read,
                      cudaStream_t                        stream);

    Coordinator& coord_;
    uint64_t     tensor_size_;
    bool         use_mla_;
    uint32_t     max_entries_;   // cached from io_engine at construction
};

} // namespace tutti

#endif // __TUTTI_ADAPTERS_KV_CACHE_IO_ADAPTER_H__
