# block_storage/ -- Tutti GpuFile (block) layer (R6).
#
# R6.1 shipped the interface + a pure resolve helper.
# R6.2 lands HostFsBackedBlockStorage + PersistentGpuFileLog +
#       a host-side end-to-end smoke.
# R6.3 lands the GPU acquire/release path + a GPU device-side
#       submit smoke (one kernel-thread-per-IO across shards).
#
# Builds:
#   tutti_block_storage              -- the runtime lib (static-style)
#   block_storage_resolve_test       -- R6.1 unit test for gpu_file_resolve
#   block_storage_smoke              -- R6.2 host-side end-to-end
#   block_storage_gpu_smoke          -- R6.3 GPU device-side end-to-end
#
# Layer dependency:
#   block_storage  -> nvme_storage   (INvmeStorage / NvmeFile)
#                  -> tutti_memory   (GpuSlotPool<NvmeFileDeviceHandle*[]>
#                                     shard-pointer-table pooling; see
#                                     memory/include/gpu_slot_pool.h)
#                  -> device_manager (Device)
#                  -> stdc++fs       (std::filesystem)
#                  -> pthread        (std::mutex)

set(TUTTI_BLOCK_STORAGE_SOURCES
    src/persistent_gpu_file_log.cpp
    src/host_fs_backed_block_storage.cpp
)

add_library(tutti_block_storage ${TUTTI_BLOCK_STORAGE_SOURCES})

target_include_directories(tutti_block_storage
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(tutti_block_storage
    PUBLIC
        tutti_nvme_storage          # INvmeStorage, NvmeFile
        tutti_memory                # GpuSlotPool
        tutti_device_manager        # Device
    PRIVATE
        CUDA::cudart                # R8: cudaMalloc/cudaMemcpy/cudaFree
                                    # for GpuFileHandle::d_shards_dev
)

set_target_properties(tutti_block_storage PROPERTIES
    OUTPUT_NAME              "tutti_block_storage"
    BUILD_RPATH              "$ORIGIN"
    INSTALL_RPATH            "$ORIGIN"
    POSITION_INDEPENDENT_CODE ON
)

# ---------------------------------------------------------------------------
# Smoke tests live in the top-level examples/ tree (R9.4).
# ---------------------------------------------------------------------------

install(TARGETS tutti_block_storage         DESTINATION "lib")
install(DIRECTORY include/ DESTINATION "include/tutti/block_storage"
        FILES_MATCHING PATTERN "*.h")
