/**
 * tiered_handle_cache_smoke.cu -- standalone correctness smoke for
 * memory/include/{host_slot_pool,gpu_slot_pool,tiered_handle_cache}.h.
 *
 * Pure memory-subsystem test: no NVMe hardware, no snvme kernel
 * module needed.  Uses a trivial POD `TestItem` instead of the real
 * NvmeFileDeviceHandle so this can run on any CUDA-capable box.
 *
 * Coverage:
 *   [1] COLD build -> L2 -> L1 (builder called exactly once)
 *   [2] L1 cache hit (builder NOT called again)
 *   [3] batch get_or_build_batch with all-cold keys (single
 *       contiguous L1 promotion path)
 *   [4] L1 eviction (tiny L1 cap) -> content still correct via L2
 *       (builder NOT called again -- proves "downgrade, not delete")
 *   [5] L2 eviction (tiny L2 cap, forced) -> content lost -> builder
 *       IS called again (proves genuine delete semantics)
 *   [6] memory accounting (l1_gpu_bytes / l2_host_bytes)
 *   [7] evict_all_from_l1 + re-resolve from L2 alone
 */

#include "../../memory/include/tiered_handle_cache.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

int g_step = 0;

#define STEP_OK(fmt, ...) do {                                                 \
    ++g_step;                                                                  \
    std::fprintf(stderr, "[ OK ] step=%-2d  " fmt "\n", g_step, ##__VA_ARGS__);\
} while (0)

#define STEP_FAIL(fmt, ...) do {                                               \
    ++g_step;                                                                  \
    std::fprintf(stderr, "[FAIL] step=%-2d  " fmt "\n", g_step, ##__VA_ARGS__);\
    std::_Exit(2);                                                             \
} while (0)

#define CUDA_OK(call) do {                                                     \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) STEP_FAIL("CUDA error: %s (%s)",                    \
                                     #call, cudaGetErrorString(_e));           \
} while (0)

struct TestItem {
    uint64_t magic;
    uint64_t key_echo;
    uint64_t seed;
};

}  // namespace

int main(int argc, char** argv) {
    int cuda_dev = (argc > 1) ? std::atoi(argv[1]) : 0;
    CUDA_OK(cudaSetDevice(cuda_dev));
    STEP_OK("cudaSetDevice(%d)", cuda_dev);

    cudaStream_t stream = nullptr;
    CUDA_OK(cudaStreamCreate(&stream));

    // ---- [1]-[3] normal-size cache: L1=8, L2=32 --------------------
    {
        tutti::TieredHandleCache<TestItem, uint64_t> cache;
        tutti::TieredHandleCache<TestItem, uint64_t>::Config cfg{};
        cfg.l1_capacity = 8;
        cfg.l2_capacity = 32;
        cfg.cuda_device = cuda_dev;
        if (!cache.init(cfg)) STEP_FAIL("cache.init(l1=8,l2=32)");
        STEP_OK("cache.init(l1=8, l2=32)");

        int build_calls = 0;
        auto builder = [&](const uint64_t& key, TestItem* out) -> bool {
            ++build_calls;
            out->magic    = 0xC0FFEEULL;
            out->key_echo = key;
            out->seed     = key * 0x9E3779B97F4A7C15ULL;
            return true;
        };

        // [1] COLD build.
        TestItem* p1 = cache.get_or_build(42, builder, stream);
        if (p1 == nullptr) STEP_FAIL("get_or_build(42) COLD");
        CUDA_OK(cudaStreamSynchronize(stream));
        TestItem got{};
        CUDA_OK(cudaMemcpy(&got, p1, sizeof(TestItem), cudaMemcpyDeviceToHost));
        if (got.magic != 0xC0FFEEULL || got.key_echo != 42)
            STEP_FAIL("COLD build content mismatch (magic=0x%llx key_echo=%llu)",
                     (unsigned long long)got.magic, (unsigned long long)got.key_echo);
        if (build_calls != 1) STEP_FAIL("build_calls=%d (want 1) after COLD", build_calls);
        STEP_OK("[1] COLD build: key=42 -> L2 -> L1, content correct, builder called once");

        // [2] L1 cache hit: same key, builder must NOT be called again.
        TestItem* p2 = cache.get_or_build(42, builder, stream);
        if (p2 != p1) STEP_FAIL("cache hit returned a different pointer (p1=%p p2=%p)",
                                (void*)p1, (void*)p2);
        if (build_calls != 1) STEP_FAIL("build_calls=%d (want 1) after L1 hit", build_calls);
        STEP_OK("[2] L1 cache hit: same pointer, builder NOT called again");

        // [3] batch, all-cold (fresh keys 100..104, 5 items -- must
        //     stay <= l1_capacity=8 for a single batch call; forces
        //     the contiguous L2+L1 batch path since none exist yet).
        //     L1 now holds {42,100..104} = 6/8, no eviction yet.
        std::vector<uint64_t> batch_keys;
        for (uint64_t k = 100; k < 105; ++k) batch_keys.push_back(k);
        std::vector<TestItem*> out_ptrs(batch_keys.size(), nullptr);
        int build_calls_before_batch = build_calls;
        bool ok = cache.get_or_build_batch(batch_keys.data(), (uint32_t)batch_keys.size(),
                                           builder, stream, out_ptrs.data());
        if (!ok) STEP_FAIL("get_or_build_batch(5 cold keys)");
        CUDA_OK(cudaStreamSynchronize(stream));
        for (std::size_t i = 0; i < batch_keys.size(); ++i) {
            if (out_ptrs[i] == nullptr) STEP_FAIL("batch out_ptrs[%zu] == nullptr", i);
            TestItem g{};
            CUDA_OK(cudaMemcpy(&g, out_ptrs[i], sizeof(TestItem), cudaMemcpyDeviceToHost));
            if (g.key_echo != batch_keys[i])
                STEP_FAIL("batch item %zu: key_echo=%llu want=%llu",
                         i, (unsigned long long)g.key_echo,
                         (unsigned long long)batch_keys[i]);
        }
        if (build_calls - build_calls_before_batch != (int)batch_keys.size())
            STEP_FAIL("batch build_calls delta=%d want=%zu",
                     build_calls - build_calls_before_batch, batch_keys.size());
        STEP_OK("[3] batch get_or_build_batch: %zu cold keys, single contiguous "
                "path, content correct, builder called exactly once per key",
                batch_keys.size());

        // [3.5] a batch that exceeds l1_capacity must be REJECTED
        // outright (not silently corrupted) -- this is the exact
        // hazard the protect-set + capacity check fixes.
        std::vector<uint64_t> too_big_batch;
        for (uint64_t k = 200; k < 210; ++k) too_big_batch.push_back(k);   // 10 > l1_capacity=8
        std::vector<TestItem*> too_big_out(too_big_batch.size(), nullptr);
        bool rejected = cache.get_or_build_batch(too_big_batch.data(),
                                                 (uint32_t)too_big_batch.size(),
                                                 builder, stream, too_big_out.data());
        if (rejected) STEP_FAIL("batch of 10 > l1_capacity=8 should have been "
                                "rejected, but get_or_build_batch returned true");
        STEP_OK("[3.5] oversized batch (10 > l1_capacity=8) correctly rejected");

        // [3.6] individually touch 3 more fresh keys (105..107) to push
        //     L1 past capacity (6 + 3 = 9 > 8) and force a real
        //     single-key LRU eviction (of key=42, the coldest).
        for (uint64_t k = 105; k < 108; ++k) {
            TestItem* p = cache.get_or_build(k, builder, stream);
            if (p == nullptr) STEP_FAIL("get_or_build(%llu) forcing eviction",
                                        (unsigned long long)k);
        }
        CUDA_OK(cudaStreamSynchronize(stream));
        STEP_OK("[3.6] touched 3 more fresh keys (105..107): L1 now over "
                "capacity, key=42 should have been LRU-evicted (downgraded)");

        // [4] L1 eviction: re-resolving key 42 (the coldest, evicted
        //     in [3.6]) must NOT call the builder again (content
        //     survives in L2) and must return correct content.
        int build_calls_before_re = build_calls;
        TestItem* p42_again = cache.get_or_build(42, builder, stream);
        if (p42_again == nullptr) STEP_FAIL("re-resolve key=42 after presumed L1 eviction");
        CUDA_OK(cudaStreamSynchronize(stream));
        TestItem g42{};
        CUDA_OK(cudaMemcpy(&g42, p42_again, sizeof(TestItem), cudaMemcpyDeviceToHost));
        if (g42.key_echo != 42 || g42.seed != (uint64_t)42 * 0x9E3779B97F4A7C15ULL)
            STEP_FAIL("re-resolved key=42 content wrong (key_echo=%llu)",
                     (unsigned long long)g42.key_echo);
        if (build_calls != build_calls_before_re)
            STEP_FAIL("builder called again for key=42 (build_calls %d -> %d) -- "
                     "L1 eviction must be a DOWNGRADE, not a delete",
                     build_calls_before_re, build_calls);
        STEP_OK("[4] L1 eviction is a downgrade: key=42 survives via L2, "
                "builder NOT re-invoked, content correct");

        // [6] memory accounting.
        const std::size_t want_l1 = (std::size_t)8  * sizeof(TestItem);
        const std::size_t want_l2 = (std::size_t)32 * sizeof(TestItem);
        if (cache.l1_gpu_bytes() != want_l1 || cache.l2_host_bytes() != want_l2)
            STEP_FAIL("memory accounting mismatch: l1=%zu(want %zu) l2=%zu(want %zu)",
                     cache.l1_gpu_bytes(), want_l1, cache.l2_host_bytes(), want_l2);
        STEP_OK("[6] memory accounting: l1_gpu_bytes=%zu l2_host_bytes=%zu",
                cache.l1_gpu_bytes(), cache.l2_host_bytes());

        // [7] evict_all_from_l1 then re-resolve everything from L2 alone.
        cache.evict_all_from_l1(stream);
        CUDA_OK(cudaStreamSynchronize(stream));
        int build_calls_before_full_reresolve = build_calls;
        for (uint64_t k : batch_keys) {
            TestItem* p = cache.get_or_build(k, builder, stream);
            if (p == nullptr) STEP_FAIL("re-resolve key=%llu after evict_all_from_l1",
                                        (unsigned long long)k);
        }
        CUDA_OK(cudaStreamSynchronize(stream));
        if (build_calls != build_calls_before_full_reresolve)
            STEP_FAIL("builder re-invoked after evict_all_from_l1 (delta=%d) -- "
                     "content should have survived in L2",
                     build_calls - build_calls_before_full_reresolve);
        STEP_OK("[7] evict_all_from_l1: all %zu batch keys re-resolved from L2 "
                "alone, zero rebuilds", batch_keys.size());

        cache.shutdown();
    }

    // ---- [8] admit(): L2-only insertion bound to open/close ---------
    {
        tutti::TieredHandleCache<TestItem, uint64_t> cache;
        tutti::TieredHandleCache<TestItem, uint64_t>::Config cfg{};
        cfg.l1_capacity = 4;
        cfg.l2_capacity = 16;
        cfg.cuda_device = cuda_dev;
        if (!cache.init(cfg)) STEP_FAIL("cache.init(l1=4,l2=16) for admit test");

        int build_calls = 0;
        auto builder = [&](const uint64_t& key, TestItem* out) -> bool {
            ++build_calls;
            out->magic = 0xC0FFEEULL; out->key_echo = key; out->seed = key;
            return true;
        };

        // admit a COLD key: builder called exactly once, entry now in
        // L2 (no GPU/stream involved).
        if (!cache.admit(500, builder)) STEP_FAIL("admit(500) COLD");
        if (build_calls != 1) STEP_FAIL("admit COLD build_calls=%d (want 1)", build_calls);

        // admit again: idempotent, builder NOT called a second time.
        if (!cache.admit(500, builder)) STEP_FAIL("admit(500) idempotent");
        if (build_calls != 1)
            STEP_FAIL("admit idempotent build_calls=%d (want 1)", build_calls);

        // get_or_build now finds it in L2 -> promotes to L1 with NO
        // rebuild (proves admit populated L2), content correct.
        TestItem* p = cache.get_or_build(500, builder, stream);
        if (p == nullptr) STEP_FAIL("get_or_build(500) after admit");
        CUDA_OK(cudaStreamSynchronize(stream));
        TestItem g{};
        CUDA_OK(cudaMemcpy(&g, p, sizeof(TestItem), cudaMemcpyDeviceToHost));
        if (g.key_echo != 500) STEP_FAIL("admit->get_or_build content (key_echo=%llu)",
                                         (unsigned long long)g.key_echo);
        if (build_calls != 1)
            STEP_FAIL("get_or_build after admit rebuilt (build_calls=%d, want 1) -- "
                     "admit must have left a usable L2 copy", build_calls);

        // erase() then get_or_build must rebuild (admit+erase are the
        // open/close pair -- erased == closed == gone from cache).
        cache.erase(500, stream);
        CUDA_OK(cudaStreamSynchronize(stream));
        TestItem* p2 = cache.get_or_build(500, builder, stream);
        if (p2 == nullptr) STEP_FAIL("get_or_build(500) after erase");
        CUDA_OK(cudaStreamSynchronize(stream));
        if (build_calls != 2)
            STEP_FAIL("get_or_build after erase build_calls=%d (want 2) -- erase "
                     "must have removed the L2 copy too", build_calls);
        STEP_OK("[8] admit: COLD build-once + idempotent + L2-backed promote "
                "(no rebuild); erase drops it (rebuild required)");

        cache.shutdown();
    }

    // ---- [9] get_or_build_batch with duplicate keys ----------------
    //   R12 A-5 / 隐患-4: duplicate keys in a batch must resolve to
    //   the same cache entry (not leak an L2 slot by overwriting
    //   index_[key] on a second build).
    {
        tutti::TieredHandleCache<TestItem, uint64_t> cache;
        tutti::TieredHandleCache<TestItem, uint64_t>::Config cfg{};
        cfg.l1_capacity = 8;
        cfg.l2_capacity = 32;
        cfg.cuda_device = cuda_dev;
        if (!cache.init(cfg)) STEP_FAIL("cache.init(l1=8,l2=32) for dedup test");

        int build_calls = 0;
        auto builder = [&](const uint64_t& key, TestItem* out) -> bool {
            ++build_calls;
            out->magic = 0xC0FFEEULL; out->key_echo = key; out->seed = key;
            return true;
        };

        // Batch with duplicate keys: [300, 301, 300, 302, 301]
        // 3 unique keys (300, 301, 302), 5 total entries.
        std::vector<uint64_t> dup_keys = {300, 301, 300, 302, 301};
        std::vector<TestItem*> dup_out(dup_keys.size(), nullptr);
        auto stats_before = cache.stats();
        bool ok = cache.get_or_build_batch(dup_keys.data(), (uint32_t)dup_keys.size(),
                                           builder, stream, dup_out.data());
        if (!ok) STEP_FAIL("get_or_build_batch with duplicate keys");
        CUDA_OK(cudaStreamSynchronize(stream));

        // (1) Each unique key built exactly once: build_calls == 3, not 5.
        if (build_calls != 3)
            STEP_FAIL("dedup build_calls=%d (want 3) -- duplicate keys must "
                     "not trigger redundant builds", build_calls);

        // (2) Duplicate keys share the same pointer.
        if (dup_out[0] != dup_out[2])
            STEP_FAIL("dup keys [0] and [2] (both=300) have different pointers "
                     "(%p vs %p)", (void*)dup_out[0], (void*)dup_out[2]);
        if (dup_out[1] != dup_out[4])
            STEP_FAIL("dup keys [1] and [4] (both=301) have different pointers "
                     "(%p vs %p)", (void*)dup_out[1], (void*)dup_out[4]);

        // (3) Content correct for all entries.
        for (std::size_t i = 0; i < dup_keys.size(); ++i) {
            if (dup_out[i] == nullptr) STEP_FAIL("dup batch out[%zu] == nullptr", i);
            TestItem g{};
            CUDA_OK(cudaMemcpy(&g, dup_out[i], sizeof(TestItem), cudaMemcpyDeviceToHost));
            if (g.key_echo != dup_keys[i])
                STEP_FAIL("dup batch item %zu: key_echo=%llu want=%llu",
                         i, (unsigned long long)g.key_echo,
                         (unsigned long long)dup_keys[i]);
        }

        // (4) Stats: cold_builds delta == unique count (3), not total (5).
        auto stats_after = cache.stats();
        if (stats_after.cold_builds - stats_before.cold_builds != 3)
            STEP_FAIL("dedup cold_builds delta=%llu (want 3)",
                     (unsigned long long)(stats_after.cold_builds - stats_before.cold_builds));

        STEP_OK("[9] get_or_build_batch dedup: 5 entries / 3 unique keys -> "
                "3 builds, dup keys share pointers, content correct");

        cache.shutdown();
    }

    // ---- [5] tiny cache to force genuine L2 eviction ---------------
    {
        tutti::TieredHandleCache<TestItem, uint64_t> cache;
        tutti::TieredHandleCache<TestItem, uint64_t>::Config cfg{};
        cfg.l1_capacity = 2;
        cfg.l2_capacity = 4;   // deliberately tiny -> forces real L2 eviction
        cfg.cuda_device = cuda_dev;
        if (!cache.init(cfg)) STEP_FAIL("cache.init(l1=2,l2=4)");

        std::vector<int> build_count(64, 0);
        auto builder = [&](const uint64_t& key, TestItem* out) -> bool {
            build_count[key] += 1;
            out->magic = 0xC0FFEEULL; out->key_echo = key; out->seed = key;
            return true;
        };

        // Fill past L2 capacity: keys 0..5 (6 keys > l2_capacity=4).
        for (uint64_t k = 0; k < 6; ++k) {
            TestItem* p = cache.get_or_build(k, builder, stream);
            if (p == nullptr) STEP_FAIL("get_or_build(%llu) in L2-overflow test",
                                        (unsigned long long)k);
        }
        CUDA_OK(cudaStreamSynchronize(stream));

        // Key 0 (coldest, and NOT pinned by L1 since L1 cap=2 and 5
        // keys got resident after it) must have been evicted from L2
        // -- re-resolving it should call the builder AGAIN.
        int before = build_count[0];
        TestItem* p0 = cache.get_or_build(0, builder, stream);
        if (p0 == nullptr) STEP_FAIL("re-resolve key=0 after presumed L2 eviction");
        CUDA_OK(cudaStreamSynchronize(stream));
        if (build_count[0] <= before)
            STEP_FAIL("key=0 builder NOT re-invoked (count=%d) -- expected a "
                     "genuine L2 eviction with tiny l2_capacity=4",
                     build_count[0]);
        STEP_OK("[5] L2 eviction is a genuine delete: key=0 required a rebuild "
                "(build_count %d -> %d) once evicted from the tiny L2",
                before, build_count[0]);

        cache.shutdown();
    }

    cudaStreamDestroy(stream);

    std::fprintf(stderr,
        "\n=== tiered_handle_cache_smoke: all %d steps passed ===\n", g_step);
    return 0;
}
