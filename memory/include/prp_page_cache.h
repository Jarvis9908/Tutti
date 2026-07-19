#ifndef __TUTTI_MEMORY_PRP_PAGE_CACHE_H__
#define __TUTTI_MEMORY_PRP_PAGE_CACHE_H__

/**
 * prp_page_cache.h -- two-tier (host-pinned L2 + GPU-DMA L1) cache for
 * PRP-list pages (R12 Phase C-1/C-2).
 *
 * Layer: memory.
 *
 * WHY THIS EXISTS (see the discussion in the R12 thread):
 *
 *   A PRP-list page is a 4 KiB GPU-resident, DMA-mapped page whose
 *   IOVA the NVMe controller fetches (via an AddressDescriptor's prp2)
 *   to learn the physical pages of a >2-page IO.  Under PageAttention,
 *   the number of registered KV-cache blocks (each needing its own
 *   PRP page) can vastly exceed what fits in GPU memory at once, even
 *   though the "hot" working set of any single IO batch is small.
 *   Keeping every block's PRP page GPU-resident would waste scarce GPU
 *   memory that must compete with model weights + the KV tensors.
 *
 *   The tensors themselves are registered ONCE at process start and
 *   live for the whole run (no high-frequency register/unregister).
 *   What must be paged in/out is only the PRP-list PAGE backing each
 *   registered IO -- exactly the shape TieredHandleCache already
 *   solves for NvmeFileDeviceHandles:
 *
 *     L2 (host-pinned, large): the PERSISTENT backing store.  Every
 *         admitted PRP page's content lives here for the tensor's
 *         whole lifetime.  Built ONCE at register_tensor time.
 *     L1 (GPU, DMA-mapped, small): the transient working set.  A page
 *         is promoted (H2D copy L2->L1) only just before an IO batch
 *         that needs it; L1-full evicts the LRU page (a DOWNGRADE --
 *         the L2 copy is untouched, so re-promotion is one memcpy).
 *
 * THE prp2 PATCH PROBLEM (why this is harder than the handle cache):
 *
 *   TieredHandleCache caches a POINTER (shards[fd_idx]); the GPU kernel
 *   dereferences it, so wherever the slot physically sits is fine.
 *   A PRP page instead contributes a RAW IOVA VALUE stored in the
 *   tensor's long-lived, GPU-resident AddressDescriptor[].prp2 -- the
 *   value IS the content, consumed directly by the controller with no
 *   indirection.  So when a page moves L2<->L1 its IOVA changes, and
 *   the owning descriptor's prp2 MUST be rewritten to the new L1 IOVA
 *   before the next IO uses it.
 *
 *   ensure_resident_batch() does this for a whole IO batch at once:
 *   it promotes every needed page, collects the (prp2-field-device-
 *   addr, new-IOVA) pairs whose IOVA actually changed, and applies
 *   them with ONE scatter kernel + ONE H2D staging copy -- never one
 *   cudaMemcpyAsync per page (the "one batch must not launch thousands
 *   of tiny copies" anti-pattern this codebase repeatedly avoids; see
 *   gpu_slot_pool.h / tiered_handle_cache.h).
 *
 * KEY: the device address &AddressDescriptor.prp2 for the owning IO.
 *   Each tensor's descriptor blob is an independent cudaMalloc, so
 *   this address is globally unique -- no region_id needed.
 *
 * EVENT FENCING (L1 slot reuse): identical shape + limitation to
 *   GpuSlotPool -- one shared cudaEvent per DISTINCT stream, re-
 *   recorded on fill/evict; a reused L1 slot waits on its previous
 *   occupant's last-touch stream event.  Correct for the codebase's
 *   "produce and consume a given page on the SAME stream" convention;
 *   conservative (never under-synchronizing) otherwise.  See
 *   gpu_slot_pool.h's file comment for the full argument.
 *
 * Thread-safety: one mutex guards the index + LRU + pool calls for the
 * duration of one admit / ensure_resident_batch / erase (single
 * control-plane thread per Coordinator, matching the rest of memory/).
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <list>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>

#include "prp_list_pool.h"

namespace tutti {

// Defined in host_device_memory_subsystem.cu (needs nvcc for <<<>>>).
// Scatters `values[i]` into `*targets[i]` -- one thread per pair.
void prp_launch_patch_prp2(uint64_t* const* d_targets,
                           const uint64_t*  d_values,
                           uint32_t         count,
                           cudaStream_t     stream);

class PrpPageCache {
public:
    using Config = PrpListPool::Config;

    struct Stats {
        uint64_t admits        = 0;  // register-time L2 inserts
        uint64_t l1_hits       = 0;  // ensure_resident: already L1-resident
        uint64_t l1_promotions = 0;  // ensure_resident: L2->L1 copy
        uint64_t l1_evictions  = 0;  // L1->L2 downgrade (GPU slot freed)
        uint64_t patches       = 0;  // prp2 fields rewritten (scatter kernel)
        uint64_t l2_exhausted  = 0;  // admit failed (no L2 slot)
        uint64_t l1_exhausted  = 0;  // ensure failed (batch > L1 capacity)
    };

    PrpPageCache() = default;
    ~PrpPageCache() { shutdown(); }

    PrpPageCache(const PrpPageCache&)            = delete;
    PrpPageCache& operator=(const PrpPageCache&) = delete;

    bool ready() const { return pool_.ready(); }
    uint32_t l1_capacity() const { return pool_.l1_capacity(); }
    uint32_t l2_capacity() const { return pool_.l2_capacity(); }
    uint32_t slot_bytes()  const { return pool_.slot_bytes(); }

    Stats stats() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return stats_;
    }

    /// One-time setup.  `cuda_device` is where L1 + the scatter-patch
    /// staging live.  Returns false on any allocation failure (caller
    /// falls back to per-tensor owned PRP buffers).
    bool init(const Config& cfg, nvm_ctrl_t* ctrl, int cuda_device) {
        if (pool_.ready()) return true;
        if (!pool_.init(cfg, ctrl)) return false;

        cuda_device_ = cuda_device;
        int prev = -1;
        cudaGetDevice(&prev);
        cudaSetDevice(cuda_device);

        const uint32_t cap = pool_.l1_capacity();  // max pages patchable/batch
        bool ok = true;
        ok = ok && cudaMalloc((void**)&d_targets_, (std::size_t)cap * sizeof(uint64_t*)) == cudaSuccess;
        ok = ok && cudaMalloc((void**)&d_values_,  (std::size_t)cap * sizeof(uint64_t))  == cudaSuccess;
        ok = ok && cudaMallocHost((void**)&h_targets_, (std::size_t)cap * sizeof(uint64_t*)) == cudaSuccess;
        ok = ok && cudaMallocHost((void**)&h_values_,  (std::size_t)cap * sizeof(uint64_t))  == cudaSuccess;
        if (!ok) {
            std::fprintf(stderr,
                "[prp_cache] init: scatter-patch staging alloc failed "
                "(cap=%u)\n", cap);
            free_staging_();
            pool_.shutdown();
            cudaSetDevice(prev);
            return false;
        }
        last_touch_stream_.assign(cap, nullptr);
        slot_ever_touched_.assign(cap, 0);
        cudaSetDevice(prev);
        std::fprintf(stderr,
            "[prp_cache] init: scatter-patch staging = %u pairs\n", cap);
        return true;
    }

    void shutdown() {
        std::lock_guard<std::mutex> lock(mtx_);
        int prev = -1;
        const bool have_dev = (d_targets_ != nullptr) || !stream_events_.empty();
        if (have_dev) { cudaGetDevice(&prev); cudaSetDevice(cuda_device_); }
        for (auto& kv : stream_events_) cudaEventDestroy(kv.second);
        stream_events_.clear();
        free_staging_();
        if (have_dev) cudaSetDevice(prev);
        pool_.shutdown();
        index_.clear();
        l1_lru_.clear();
        l1_lru_pos_.clear();
        last_touch_stream_.clear();
        slot_ever_touched_.clear();
        stats_ = {};
    }

    /// Register-time: copy `page_content_host` (slot_bytes bytes) into
    /// a fresh L2 slot and record the entry keyed by `prp2_devptr`
    /// (= &AddressDescriptor.prp2, a device address).  Does NOT touch
    /// L1: promotion is deferred to the first ensure_resident_batch.
    /// Returns false if L2 is exhausted (caller falls back to an owned
    /// always-resident PRP buffer for the whole tensor).
    bool admit(uint64_t* prp2_devptr, const void* page_content_host) {
        std::lock_guard<std::mutex> lock(mtx_);
        const uint64_t key = reinterpret_cast<uint64_t>(prp2_devptr);
        if (index_.find(key) != index_.end()) return true;  // idempotent

        uint32_t l2_slot = 0;
        void* l2ptr = pool_.acquire_l2(&l2_slot);
        if (l2ptr == nullptr) { ++stats_.l2_exhausted; return false; }
        std::memcpy(l2ptr, page_content_host, pool_.slot_bytes());

        Entry e;
        e.prp2_devptr   = prp2_devptr;
        e.l2_slot       = l2_slot;
        e.l1_slot       = UINT32_MAX;
        e.resident      = false;
        e.patched_ioaddr = UINT64_MAX;   // force first patch
        index_.emplace(key, e);
        ++stats_.admits;
        return true;
    }

    /// IO-time: make every page in `prp2_devptrs[0..count)` L1-resident
    /// and rewrite each descriptor's prp2 to its (possibly new) L1 IOVA
    /// via ONE scatter kernel on `stream`.  All `count` pages must fit
    /// in L1 for the duration of the batch (they are protected from
    /// eviction here); returns false if the batch exceeds L1 capacity
    /// or any key was never admitted.
    bool ensure_resident_batch(uint64_t* const* prp2_devptrs,
                               uint32_t         count,
                               cudaStream_t     stream) {
        if (count == 0) return true;
        std::lock_guard<std::mutex> lock(mtx_);
        if (count > pool_.l1_capacity()) {
            std::fprintf(stderr,
                "[prp_cache] ensure_resident_batch: count=%u > l1_capacity=%u "
                "(batch working set must fit in L1)\n", count, pool_.l1_capacity());
            ++stats_.l1_exhausted;
            return false;
        }

        std::unordered_set<uint64_t> protect;
        protect.reserve(count);
        for (uint32_t i = 0; i < count; ++i)
            protect.insert(reinterpret_cast<uint64_t>(prp2_devptrs[i]));

        uint32_t patch_n = 0;
        for (uint32_t i = 0; i < count; ++i) {
            const uint64_t key = reinterpret_cast<uint64_t>(prp2_devptrs[i]);
            auto it = index_.find(key);
            if (it == index_.end()) {
                std::fprintf(stderr,
                    "[prp_cache] ensure_resident_batch: key %p never admitted\n",
                    (void*)prp2_devptrs[i]);
                return false;
            }
            Entry& e = it->second;

            if (e.resident) {
                touch_l1_lru_locked_(key);
                ++stats_.l1_hits;
                continue;   // prp2 already patched to this slot's IOVA
            }

            // Promote L2 -> L1 (evict LRU if full, never our own batch).
            uint32_t l1_slot = 0, ioaddr_base = 0;
            void* l1ptr = pool_.acquire_l1(&l1_slot, &ioaddr_base);
            while (l1ptr == nullptr) {
                if (!evict_l1_lru_one_locked_(stream, protect)) {
                    std::fprintf(stderr,
                        "[prp_cache] ensure_resident_batch: L1 exhausted with "
                        "nothing evictable (batch too large?)\n");
                    ++stats_.l1_exhausted;
                    return false;
                }
                l1ptr = pool_.acquire_l1(&l1_slot, &ioaddr_base);
            }

            // Reuse fence: a recycled L1 slot may still be mid-DMA for a
            // prior batch's IO on last_touch_stream_[l1_slot].
            if (slot_ever_touched_[l1_slot]) {
                cudaEvent_t ev = event_for_stream_(last_touch_stream_[l1_slot]);
                if (ev != nullptr) cudaStreamWaitEvent(stream, ev, 0);
            }

            cudaError_t cerr = cudaMemcpyAsync(
                l1ptr, pool_.l2_slot_ptr(e.l2_slot), pool_.slot_bytes(),
                cudaMemcpyHostToDevice, stream);
            if (cerr != cudaSuccess) {
                std::fprintf(stderr,
                    "[prp_cache] ensure_resident_batch: H2D promote failed: %s\n",
                    cudaGetErrorString(cerr));
                pool_.release_l1(l1_slot);
                return false;
            }
            last_touch_stream_[l1_slot] = stream;
            slot_ever_touched_[l1_slot] = 1;

            e.l1_slot  = l1_slot;
            e.resident = true;
            l1_lru_.push_front(key);
            l1_lru_pos_[key] = l1_lru_.begin();
            ++stats_.l1_promotions;

            const uint64_t new_ioaddr = pool_.l1_ioaddr(l1_slot);
            if (e.patched_ioaddr != new_ioaddr) {
                h_targets_[patch_n] = e.prp2_devptr;
                h_values_[patch_n]  = new_ioaddr;
                ++patch_n;
                e.patched_ioaddr = new_ioaddr;
            }
        }

        // Record this stream's shared event AFTER the promote copies so a
        // future reuse of any slot we just filled fences behind them.
        cudaEvent_t ev = event_for_stream_(stream);
        if (ev != nullptr) cudaEventRecord(ev, stream);

        // ONE scatter kernel patches every changed prp2 (never per-page).
        if (patch_n > 0) {
            cudaMemcpyAsync(d_targets_, h_targets_,
                            (std::size_t)patch_n * sizeof(uint64_t*),
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_values_, h_values_,
                            (std::size_t)patch_n * sizeof(uint64_t),
                            cudaMemcpyHostToDevice, stream);
            prp_launch_patch_prp2(d_targets_, d_values_, patch_n, stream);
            stats_.patches += patch_n;
        }
        return true;
    }

    /// Free-time: drop the cache entries for `prp2_devptrs[0..count)`
    /// (releases their L1 slots -- event-fenced on `stream` -- and L2
    /// backing slots).  Missing keys are ignored.
    void erase(uint64_t* const* prp2_devptrs, uint32_t count, cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(mtx_);
        for (uint32_t i = 0; i < count; ++i) {
            const uint64_t key = reinterpret_cast<uint64_t>(prp2_devptrs[i]);
            auto it = index_.find(key);
            if (it == index_.end()) continue;
            Entry& e = it->second;
            if (e.resident) {
                cudaEvent_t ev = event_for_stream_(stream);
                if (ev != nullptr) cudaEventRecord(ev, stream);
                last_touch_stream_[e.l1_slot] = stream;
                slot_ever_touched_[e.l1_slot] = 1;
                pool_.release_l1(e.l1_slot);
                auto lit = l1_lru_pos_.find(key);
                if (lit != l1_lru_pos_.end()) { l1_lru_.erase(lit->second); l1_lru_pos_.erase(lit); }
            }
            pool_.release_l2(e.l2_slot);
            index_.erase(it);
        }
    }

private:
    struct Entry {
        uint64_t* prp2_devptr    = nullptr;  // &AddressDescriptor.prp2 (device)
        uint32_t  l2_slot        = UINT32_MAX;
        uint32_t  l1_slot        = UINT32_MAX;   // valid iff resident
        bool      resident       = false;
        uint64_t  patched_ioaddr = UINT64_MAX;   // last IOVA written to prp2
    };

    void touch_l1_lru_locked_(uint64_t key) {
        auto it = l1_lru_pos_.find(key);
        if (it != l1_lru_pos_.end()) l1_lru_.erase(it->second);
        l1_lru_.push_front(key);
        l1_lru_pos_[key] = l1_lru_.begin();
    }

    // Evict the LRU-tail page NOT in `protect`.  DOWNGRADE only: the L2
    // copy stays, so the descriptor's prp2 is now stale and its entry is
    // marked non-resident (next ensure re-promotes + re-patches).
    bool evict_l1_lru_one_locked_(cudaStream_t stream,
                                  const std::unordered_set<uint64_t>& protect) {
        for (auto it = l1_lru_.rbegin(); it != l1_lru_.rend(); ++it) {
            if (protect.count(*it) != 0) continue;
            const uint64_t victim = *it;
            Entry& e = index_.at(victim);
            cudaEvent_t ev = event_for_stream_(stream);
            if (ev != nullptr) cudaEventRecord(ev, stream);
            last_touch_stream_[e.l1_slot] = stream;
            slot_ever_touched_[e.l1_slot] = 1;
            pool_.release_l1(e.l1_slot);
            e.resident = false;
            e.l1_slot  = UINT32_MAX;
            // NOTE: patched_ioaddr kept -- if the slot happens to come
            // back with the same IOVA a re-patch is skipped; otherwise
            // the mismatch forces a re-patch.  Either way prp2 is fixed
            // before the page is used again (it is non-resident now, so
            // ensure_resident_batch will promote+patch first).
            auto pit = l1_lru_pos_.find(victim);
            l1_lru_.erase(pit->second);
            l1_lru_pos_.erase(pit);
            ++stats_.l1_evictions;
            return true;
        }
        return false;
    }

    cudaEvent_t event_for_stream_(cudaStream_t stream) {
        auto it = stream_events_.find(stream);
        if (it != stream_events_.end()) return it->second;
        cudaEvent_t e = nullptr;
        if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) {
            std::fprintf(stderr, "[prp_cache] event_for_stream_: create failed\n");
            return nullptr;
        }
        stream_events_.emplace(stream, e);
        return e;
    }

    void free_staging_() {
        if (d_targets_) { cudaFree(d_targets_); d_targets_ = nullptr; }
        if (d_values_)  { cudaFree(d_values_);  d_values_  = nullptr; }
        if (h_targets_) { cudaFreeHost(h_targets_); h_targets_ = nullptr; }
        if (h_values_)  { cudaFreeHost(h_values_);  h_values_  = nullptr; }
    }

    mutable std::mutex mtx_;
    PrpListPool        pool_;
    int                cuda_device_ = 0;
    Stats              stats_{};

    std::unordered_map<uint64_t, Entry> index_;   // prp2_devptr -> Entry
    std::list<uint64_t> l1_lru_;                   // MRU front; L1-resident keys
    std::unordered_map<uint64_t, std::list<uint64_t>::iterator> l1_lru_pos_;

    // Scatter-patch staging (sized to L1 capacity).
    uint64_t** d_targets_ = nullptr;   // device: prp2 field addresses
    uint64_t*  d_values_  = nullptr;   // device: new IOVAs
    uint64_t** h_targets_ = nullptr;   // pinned host mirror
    uint64_t*  h_values_  = nullptr;

    // Per-L1-slot last-touch stream + shared per-stream events (mirror
    // of GpuSlotPool's shared-event scheme -- see gpu_slot_pool.h).
    std::vector<cudaStream_t> last_touch_stream_;   // l1_capacity entries
    std::vector<uint8_t>      slot_ever_touched_;
    std::unordered_map<cudaStream_t, cudaEvent_t> stream_events_;
};

} // namespace tutti

#endif // __TUTTI_MEMORY_PRP_PAGE_CACHE_H__
