# device_manager/ -- Tutti device manager (v0.1)
#
# Builds: libtutti_device_manager.so + registry_smoke
#
# Two IDeviceRegistry implementations:
#   - LocalNvmeDirectRegistry        (IN_PROCESS bootstrap)
#   - NvmeServiceBackedRegistry      (SERVICE_CLIENT bootstrap)
#
# Plus:
#   - NvmeQueueGroup                 (per-fd NVMe user queue pool +
#                                     GPU-resident d_qps[])
#
# Depends on:
#   libnvm        -- nvm_controller_init_b3, nvm_ctrl_attach_client,
#                    nvm_create_group / add_user_queue / destroy_group
#   nvmeservice   -- NvmeServiceClient (gRPC client for service mode)
#   CUDA::cudart  -- cudaMalloc / cudaMemcpy / cudaSetDevice
#                    (for NvmeQueueGroup d_qps + smoke)

set(TUTTI_DEVICE_MANAGER_SOURCES
    src/local_nvme_direct_registry.cpp
    src/nvmeservice_backed_registry.cpp
    src/nvme_queue_group.cu                # R5b
)

add_library(tutti_device_manager ${TUTTI_DEVICE_MANAGER_SOURCES})

target_include_directories(tutti_device_manager
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${PROJECT_SOURCE_DIR}/coordinator/include"          # for coordinator/device.h
        "${PROJECT_SOURCE_DIR}/io_engine/include"        # for backend_type.h, capability_set.h
    PRIVATE
        "${PROJECT_SOURCE_DIR}/backends/local/nvme/libnvm/include"
        "${PROJECT_SOURCE_DIR}/backends/local/NVMeService/src"
)

target_link_libraries(tutti_device_manager
    PUBLIC
        libnvm
    PRIVATE
        nvmeservice
        CUDA::cudart                       # R5b: NvmeQueueGroup uses cudaMalloc/Memcpy
)

set_target_properties(tutti_device_manager PROPERTIES
    OUTPUT_NAME              "tutti_device_manager"
    BUILD_RPATH              "$ORIGIN"
    INSTALL_RPATH            "$ORIGIN"
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Smoke tests live in the top-level examples/ tree.

install(TARGETS tutti_device_manager DESTINATION "lib")
install(DIRECTORY include/ DESTINATION "include/tutti/device_manager"
        FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh")
