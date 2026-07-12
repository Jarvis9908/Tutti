# coordinator/ -- Tutti top-level orchestrator (R9).
#
# Owns one instance of every layer below it and sequences bring-up /
# tear-down:
#   device_manager -> nvme_storage -> block_storage -> memory -> io_engine
#
# Ships:
#   tutti_coordinator   : the Coordinator class (bootstrap / shutdown +
#                         high-level create_gpu_file / register_tensor /
#                         submit_batch passthroughs).
#
# The runtime-visible noun headers (device.h / capability_set.h /
# lease.h) live in coordinator/include/ as header-only types shared
# with device_manager; they carry no compiled unit.

set(TUTTI_COORDINATOR_SOURCES
    src/coordinator.cu
)

add_library(tutti_coordinator ${TUTTI_COORDINATOR_SOURCES})

target_include_directories(tutti_coordinator
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(tutti_coordinator
    PUBLIC
        tutti_memory                # IMemorySubsystem, DescriptorFormat
        tutti_block_storage         # GpuFile / GpuFileHandle / IBlockStorage
        tutti_nvme_storage          # INvmeStorage
        tutti_device_manager        # Local/Service registries, Device
        tutti_io_engine_local_nvme  # IIoEngine, LocalNvmeIoEngine
    PRIVATE
        libnvm
        CUDA::cudart
)

set_target_properties(tutti_coordinator PROPERTIES
    OUTPUT_NAME              "tutti_coordinator"
    BUILD_RPATH              "$ORIGIN"
    INSTALL_RPATH            "$ORIGIN"
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

install(TARGETS tutti_coordinator DESTINATION "lib")
install(DIRECTORY include/ DESTINATION "include/tutti/coordinator"
        FILES_MATCHING PATTERN "*.h")
