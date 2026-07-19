# memory/ -- Tutti memory subsystem (v0.1)
#
# Builds: libtutti_memory.so + memory_smoke
#
# - libtutti_memory: HostDeviceMemorySubsystem (CUDA + libnvm-dependent)
# - memory_smoke   : end-to-end smoke test against real libnvm + GPU
#
# Depends on:
#   libnvm  -- nvm_dma_map_data_*, nvm_controller_init_b3
#   CUDA    -- cudart

set(TUTTI_MEMORY_SOURCES
    src/host_device_memory_subsystem.cu
)

add_library(tutti_memory ${TUTTI_MEMORY_SOURCES})

target_include_directories(tutti_memory
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
        "${PROJECT_SOURCE_DIR}/backends/local/nvme/libnvm/include"
)

target_link_libraries(tutti_memory
    PUBLIC
        libnvm
        CUDA::cudart
)

set_target_properties(tutti_memory PROPERTIES
    OUTPUT_NAME "tutti_memory"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    BUILD_RPATH   "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)

# Smoke tests live in the top-level examples/ tree.

install(TARGETS tutti_memory DESTINATION "lib")
install(DIRECTORY include/ DESTINATION "include/tutti/memory"
        FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh")
