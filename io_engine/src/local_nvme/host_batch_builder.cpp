/**
 * host_batch_builder.cpp -- implementation of build_nvme_batch (R8.1).
 *
 * Pure host code; no CUDA runtime calls (we only store GPU pointers
 * by value, never dereference them).  Keep as .cpp so the compile
 * unit doesn't pull in nvcc unnecessarily.
 */

#include "host_batch_builder.h"

#include "nvme_batch.h"

#include "memory_subsystem.h"        // IMemorySubsystem, IoSliceView
#include "block_storage.h"           // GpuFileHandle

#include <cstdio>
#include <cstdint>

namespace tutti {

bool build_nvme_batch(IMemorySubsystem*                        mem,
                      const std::vector<NvmeBatchInputTensor>& inputs,
                      bool                                     is_read,
                      NvmeBatchEntry*                          out_entries,
                      uint32_t                                 out_capacity,
                      uint32_t*                                out_count)
{
    if (mem == nullptr || out_entries == nullptr || out_count == nullptr) {
        std::fprintf(stderr,
            "[io_engine/build_nvme_batch] null arg: mem=%p out=%p count=%p\n",
            (const void*)mem, (const void*)out_entries, (const void*)out_count);
        if (out_count != nullptr) *out_count = 0;
        return false;
    }

    uint32_t k = 0;

    for (std::size_t t = 0; t < inputs.size(); ++t) {
        const NvmeBatchInputTensor& in = inputs[t];

        if (in.tensor_region == nullptr) {
            std::fprintf(stderr,
                "[io_engine/build_nvme_batch] inputs[%zu].tensor_region == nullptr\n",
                t);
            *out_count = k;
            return false;
        }
        if (in.file_handle == nullptr) {
            std::fprintf(stderr,
                "[io_engine/build_nvme_batch] inputs[%zu].file_handle == nullptr\n",
                t);
            *out_count = k;
            return false;
        }
        if (in.file_handle->d_shards_dev == nullptr ||
            in.file_handle->num_shards   == 0) {
            std::fprintf(stderr,
                "[io_engine/build_nvme_batch] inputs[%zu].file_handle has no "
                "d_shards_dev (R8 requires acquire_device_handle to populate it; "
                "num_shards=%u, d_shards_dev=%p)\n",
                t, in.file_handle->num_shards,
                (const void*)in.file_handle->d_shards_dev);
            *out_count = k;
            return false;
        }

        // R7 list_io_slices: returns the per-tensor IoSliceView[] in
        // slice_addr order.  Empty -> the region was registered without
        // granularity, which R8 does not support.
        std::vector<IoSliceView> views =
            mem->list_io_slices(in.tensor_region);
        if (views.empty()) {
            std::fprintf(stderr,
                "[io_engine/build_nvme_batch] inputs[%zu]: list_io_slices() "
                "empty -- register_tensor() with granularity > 0 is required\n",
                t);
            *out_count = k;
            return false;
        }

        // Tensor-global sub-slice running index.  Legacy semantics
        // (the pre-refactor fill_ctx loops cache_mappings.size() times,
        //  with j as the global index inside that PRPMappingEntrySpan).
        uint32_t prp_idx = 0;

        for (std::size_t s = 0; s < views.size(); ++s) {
            const IoSliceView& v = views[s];

            if (v.d_ios == nullptr || v.num_ios == 0) {
                std::fprintf(stderr,
                    "[io_engine/build_nvme_batch] inputs[%zu] view[%zu]: "
                    "d_ios=%p num_ios=%u (R7 invariant violation)\n",
                    t, s, (const void*)v.d_ios, v.num_ios);
                *out_count = k;
                return false;
            }

            for (uint32_t sub = 0; sub < v.num_ios; ++sub) {
                if (k >= out_capacity) {
                    std::fprintf(stderr,
                        "[io_engine/build_nvme_batch] out_capacity=%u "
                        "exhausted while writing inputs[%zu] view[%zu] "
                        "sub=%u (need at least %u more entries)\n",
                        out_capacity, t, s, sub, 1);
                    *out_count = k;
                    return false;
                }

                NvmeBatchEntry& e = out_entries[k];
                e.shards      = in.file_handle->d_shards_dev;
                e.num_shards  = in.file_handle->num_shards;
                // v.d_ios points to a GPU-resident contiguous
                // AddressDescriptor[]; computing &v.d_ios[sub] is
                // address arithmetic only, no host-side dereference.
                e.prp_entry   = v.d_ios + sub;
                e.prp_idx     = prp_idx;
                e.file_offset = in.file_byte_offset;
                e.is_read     = is_read;
                for (int p = 0; p < 7; ++p) e.pad[p] = 0;

                ++k;
                ++prp_idx;
            }
        }
    }

    *out_count = k;
    return true;
}

} // namespace tutti
