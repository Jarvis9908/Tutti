#ifndef __TUTTI_MEMORY_PRP_LIST_POOL_H__
#define __TUTTI_MEMORY_PRP_LIST_POOL_H__

/**
 * prp_list_pool.h -- two-tier PRP-list byte-slot allocator (R12 Phase C-1).
 *
 * Eliminates the per-tensor 64 KiB alignment waste in
 * dma_alloc_device_data: instead of cudaMalloc + nvm_dma_map_data_device
 * per tensor (which forces 64 KiB vaddr alignment + size round-up per
 * allocation), the pool does ONE cudaMalloc + ONE nvm_dma_map_data_device
 * at init time and hands out fixed-size slots.
 *
 * This class is a DUMB byte-slot allocator only -- O(1) acquire/release
 * via a bump region + free-list (same shape as GpuSlotPool/HostSlotPool).
 * It has NO knowledge of streams, events, or cache semantics (LRU,
 * admit, promote).  That layer lives in prp_page_cache.h (PrpPageCache),
 * which owns one PrpListPool instance and adds the tiered-cache
 * behaviour on top -- see that header for why a separate layer is
 * needed (GPU-DMA-mapped L1 residency requires event-fenced release
 * before a slot can be safely reused while an NVMe controller may
 * still be mid-DMA against its old content).
 *
 * Two tiers:
 *
 *   L1 (GPU-resident, DMA-mapped):
 *     One cudaMalloc + one nvm_dma_map_data_device for the entire pool.
 *     Slots are GPU device pointers with valid ioaddrs for NVMe DMA.
 *     Capacity = l1_budget_bytes / slot_bytes.
 *
 *   L2 (host-pinned staging / backing store):
 *     One cudaMallocHost for the entire pool.  Slots are host pointers.
 *     Capacity = l2_budget_bytes / slot_bytes.
 *
 * Default budgets: L1 = 64 MiB, L2 = 1 GiB.
 * With slot_bytes = 4 KiB (one PRP page): L1 = 16384 slots, L2 = 262144 slots.
 *
 * Thread safety: all public methods are protected by an internal mutex.
 */

#include <cstdint>
#include <cstdio>
#include <deque>
#include <mutex>
#include <vector>

#include <nvm_types.h>   // nvm_ctrl_t, nvm_dma_t

namespace tutti {

class PrpListPool {
public:
    struct Config {
        /// L1 (GPU-resident, DMA-mapped) budget in bytes.
        /// Capacity = l1_budget_bytes / slot_bytes.
        /// Default 64 MiB -> 16384 slots at 4 KiB/slot.
        uint64_t l1_budget_bytes = 64ull * 1024 * 1024;

        /// L2 (host-pinned staging) budget in bytes.
        /// Capacity = l2_budget_bytes / slot_bytes.
        /// Default 1 GiB -> 262144 slots at 4 KiB/slot.
        uint64_t l2_budget_bytes = 1ull * 1024 * 1024 * 1024;

        /// Bytes per slot.  Default = 4 KiB (one PRP page), covering
        /// the common case (128 KiB tensor -> total_ios=1 -> 1 PRP page).
        uint32_t slot_bytes = 4096;
    };

    struct Stats {
        uint64_t l1_acquires  = 0;
        uint64_t l1_releases  = 0;
        uint64_t l2_acquires  = 0;
        uint64_t l2_releases  = 0;
        uint64_t l1_exhausted = 0;  // acquire_l1 found no free slot
        uint64_t l2_exhausted = 0;  // acquire_l2 found no free slot
    };

    PrpListPool() = default;
    ~PrpListPool();

    PrpListPool(const PrpListPool&)            = delete;
    PrpListPool& operator=(const PrpListPool&) = delete;

    /// Allocate L1 (GPU) + L2 (host) pools.  Must be called once
    /// after bind_devices (which provides the ctrl).  Returns false
    /// on allocation failure.
    bool init(const Config& cfg, nvm_ctrl_t* ctrl);

    /// Release both pools.  Idempotent.
    void shutdown();

    /// Acquire an L1 slot (GPU-resident, DMA-mapped).  O(1) (bump
    /// region, or free-list pop).  On success returns the slot's GPU
    /// device pointer and sets *out_slot_index + *out_ioaddr_base.
    /// Returns nullptr if L1 is exhausted (caller should evict via
    /// the owning PrpPageCache's LRU, not retry here).
    void* acquire_l1(uint32_t* out_slot_index, uint32_t* out_ioaddr_base);

    /// Acquire an L2 slot (host-pinned, persistent backing store).
    /// O(1).  Returns the slot's host pointer, or nullptr if L2 is
    /// exhausted.
    void* acquire_l2(uint32_t* out_slot_index);

    /// Release an L1 slot back to the free list.  CPU-side
    /// IMMEDIATE -- the caller (PrpPageCache) is responsible for any
    /// GPU-side event fencing BEFORE calling this (this class has no
    /// stream/event awareness).
    void release_l1(uint32_t slot_index);

    /// Release an L2 slot back to the free list.
    void release_l2(uint32_t slot_index);

    /// Raw pointer accessors -- valid for any index in
    /// [0, capacity), regardless of current free/acquired state
    /// (mirrors GpuSlotPool::slot_ptr / HostSlotPool::slot_ptr).
    void* l1_slot_ptr(uint32_t slot_index) const {
        return static_cast<uint8_t*>(l1_devptr_) + (std::size_t)slot_index * cfg_.slot_bytes;
    }
    void* l2_slot_ptr(uint32_t slot_index) const {
        return static_cast<uint8_t*>(l2_hostptr_) + (std::size_t)slot_index * cfg_.slot_bytes;
    }

    /// IOVA of the first page of L1 slot `slot_index` (valid to feed
    /// into an AddressDescriptor::prp2).  Slots are always exactly
    /// one page today (slot_bytes == page_size, enforced at init by
    /// the caller passing page_size as slot_bytes); this holds for
    /// >1-page slots too as long as ioaddrs are contiguous per slot.
    uint64_t l1_ioaddr(uint32_t slot_index) const {
        const uint32_t ioaddrs_per_slot =
            cfg_.slot_bytes / (uint32_t)l1_dma_->page_size;
        return l1_dma_->ioaddrs[(std::size_t)slot_index * ioaddrs_per_slot];
    }

    // --- Accessors ---

    nvm_dma_t* l1_dma()      const { return l1_dma_; }
    void*      l1_devptr()   const { return l1_devptr_; }
    uint32_t   l1_capacity() const { return l1_capacity_; }
    uint32_t   l2_capacity() const { return l2_capacity_; }
    uint32_t   slot_bytes()  const { return cfg_.slot_bytes; }
    bool       ready()       const { return l1_dma_ != nullptr; }

    Stats stats() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return stats_;
    }

private:
    Config       cfg_{};
    nvm_ctrl_t*  ctrl_ = nullptr;

    // L1: GPU-resident, DMA-mapped.  Bump allocator + free-list, O(1).
    nvm_dma_t*   l1_dma_    = nullptr;
    void*        l1_raw_    = nullptr;   // cudaMalloc return
    void*        l1_devptr_ = nullptr;   // 64 KiB-aligned base
    uint32_t     l1_capacity_ = 0;
    uint32_t     l1_next_bump_ = 0;
    std::deque<uint32_t> l1_free_list_;

    // L2: host-pinned staging.  Bump allocator + free-list, O(1).
    void*        l2_hostptr_ = nullptr;  // cudaMallocHost return
    uint32_t     l2_capacity_ = 0;
    uint32_t     l2_next_bump_ = 0;
    std::deque<uint32_t> l2_free_list_;

    mutable std::mutex mtx_;
    Stats        stats_{};
};

} // namespace tutti

#endif // __TUTTI_MEMORY_PRP_LIST_POOL_H__
