# examples/ -- Tutti end-to-end examples / smokes.
#
# Every per-layer smoke (previously living under each
# subsystem's test/ directory) is collapsed into this single flat tree.  Each
# subsystem CMakeLists now only builds its library + installs headers;
# all executables are declared here.
#
# Sources include their dependencies through the linked libraries'
# PUBLIC include directories (bare header names) plus single-level
# "../<layer>/include/..." paths for the device_manager / coordinator
# noun headers that are not exposed as a target include dir.

# ---------------------------------------------------------------------------
# Helper: declare a CUDA smoke executable that links the common stack
# bits.  Usage:
#   tutti_add_cuda_smoke(<target> <source> LIB1 LIB2 ...)
# ---------------------------------------------------------------------------
function(tutti_add_cuda_smoke target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE ${ARGN})
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION  ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        BUILD_RPATH   "$ORIGIN/../lib"
        INSTALL_RPATH "$ORIGIN/../lib"
    )
    install(TARGETS ${target} DESTINATION "bin")
endfunction()

# ---- device_manager -------------------------------------------------------
tutti_add_cuda_smoke(registry_smoke device_manager/registry_smoke.cu
    tutti_device_manager libnvm CUDA::cudart)

# ---- memory ---------------------------------------------------------------
tutti_add_cuda_smoke(memory_smoke memory/memory_smoke.cu
    tutti_memory tutti_device_manager libnvm CUDA::cudart)

# Two-tier (CPU pinned L2 + GPU L1) handle cache correctness
# smoke.  Pure memory-subsystem test (no NVMe hardware needed) --
# only depends on tutti_memory (header-only gpu_slot_pool /
# host_slot_pool / tiered_handle_cache templates) + CUDA runtime.
tutti_add_cuda_smoke(tiered_handle_cache_smoke memory/tiered_handle_cache_smoke.cu
    tutti_memory CUDA::cudart)

# ---- nvme_storage ---------------------------------------------------------
tutti_add_cuda_smoke(nvme_storage_smoke nvme_storage/nvme_storage_smoke.cu
    tutti_nvme_storage tutti_device_manager libnvm CUDA::cudart)
tutti_add_cuda_smoke(nvme_storage_gpu_smoke nvme_storage/nvme_storage_gpu_smoke.cu
    tutti_nvme_storage tutti_device_manager libnvm CUDA::cudart)
tutti_add_cuda_smoke(nvme_storage_bulk_smoke nvme_storage/nvme_storage_bulk_smoke.cu
    tutti_nvme_storage tutti_device_manager libnvm CUDA::cudart)
tutti_add_cuda_smoke(nvme_storage_e2e_stress nvme_storage/nvme_storage_e2e_stress.cu
    tutti_nvme_storage tutti_device_manager libnvm CUDA::cudart)

# ---- block_storage --------------------------------------------------------
# The gpu_file_resolve unit test is plain host C++ (no CUDA); declare it directly.
add_executable(block_storage_resolve_test block_storage/gpu_file_resolve_test.cpp)
target_link_libraries(block_storage_resolve_test PRIVATE tutti_block_storage)
set_target_properties(block_storage_resolve_test PROPERTIES
    BUILD_RPATH   "$ORIGIN/../lib"
    INSTALL_RPATH "$ORIGIN/../lib"
)
install(TARGETS block_storage_resolve_test DESTINATION "bin")

tutti_add_cuda_smoke(block_storage_smoke block_storage/block_storage_smoke.cu
    tutti_block_storage tutti_nvme_storage tutti_device_manager
    libnvm CUDA::cudart)
tutti_add_cuda_smoke(block_storage_gpu_smoke block_storage/block_storage_gpu_smoke.cu
    tutti_block_storage tutti_nvme_storage tutti_device_manager
    libnvm CUDA::cudart)
tutti_add_cuda_smoke(block_storage_gpu_bw_smoke block_storage/block_storage_gpu_bw_smoke.cu
    tutti_block_storage tutti_nvme_storage tutti_device_manager
    libnvm CUDA::cudart)

# ---- io_engine ------------------------------------------------------------
tutti_add_cuda_smoke(io_engine_host_smoke io_engine/io_engine_host_smoke.cu
    tutti_io_engine_local_nvme tutti_block_storage tutti_nvme_storage
    tutti_memory tutti_device_manager libnvm CUDA::cudart)
tutti_add_cuda_smoke(io_engine_smoke io_engine/io_engine_smoke.cu
    tutti_io_engine_local_nvme tutti_block_storage tutti_nvme_storage
    tutti_memory tutti_device_manager libnvm CUDA::cudart)

# ---- coordinator (end-to-end) ----------------------------------------------
# Coordinator-driven byte-verified roundtrip across the whole stack.
tutti_add_cuda_smoke(e2e_smoke coordinator/e2e_smoke.cu
    tutti_coordinator tutti_block_storage tutti_nvme_storage
    tutti_memory tutti_device_manager libnvm CUDA::cudart)

# Large-scale metadata cache stress (1M-file batch create/read/write/delete).
tutti_add_cuda_smoke(metadata_cache_stress coordinator/metadata_cache_stress.cu
    tutti_coordinator tutti_block_storage tutti_nvme_storage
    tutti_memory tutti_device_manager libnvm CUDA::cudart)

# ---- adapters/kv_cache (KV-cache semantics on top of io_engine) -----------
# Exercises KvCacheIoAdapter (per-layer K/V offset + auto-chunking)
# end-to-end through the Coordinator.  Same binary covers both bring-up
# modes: default IN_PROCESS, or --service <endpoint> for SERVICE_CLIENT.
tutti_add_cuda_smoke(kv_cache_adapter_smoke adapters/kv_cache_adapter_smoke.cu
    tutti_kv_adapter tutti_coordinator tutti_block_storage tutti_nvme_storage
    tutti_memory tutti_device_manager libnvm CUDA::cudart)

# Large-scale KV-cache end-to-end stress (1M-file create, GPU R/W via
# KvCacheIoAdapter, L1/L2 cache eviction, crash consistency, bulk delete).
tutti_add_cuda_smoke(kv_cache_e2e_stress adapters/kv_cache_e2e_stress.cu
    tutti_kv_adapter tutti_coordinator tutti_block_storage tutti_nvme_storage
    tutti_memory tutti_device_manager libnvm CUDA::cudart)

# ---- phoenix conflict diagnostic ------------------------------------------
# Standalone diagnostic: tests nvm_dma_map_data_device (snvme.ko ->
# nvidia_p2p_get_pages) on a target GPU to pinpoint whether Phoenix's
# phxfs.ko BAR remap blocks GPU memory registration.  Links only libnvm
# + CUDA (no tutti layers needed).
tutti_add_cuda_smoke(phoenix_conflict_diag phoenix_conflict_diag.cu
    libnvm CUDA::cudart)

# ---- tutti_daemon (SERVICE_CLIENT owner half) -----------------------------
# The owner-side daemon: brings up the NVMe controllers via libnvm B3
# and serves gRPC so SERVICE_CLIENT Coordinators can attach.  Thin
# wrapper over nvmeservice::ServiceState; links the gRPC/protobuf stack
# (transitively PUBLIC via the nvmeservice target).  Plain host C++ --
# no CUDA device code -- so it does not go through tutti_add_cuda_smoke.
add_executable(tutti_daemon tutti_daemon.cpp)
target_link_libraries(tutti_daemon PRIVATE
    nvmeservice
    libnvm
    gRPC::grpc++
    protobuf::libprotobuf
    Threads::Threads
    CUDA::cudart
)
set_target_properties(tutti_daemon PROPERTIES
    BUILD_RPATH   "$ORIGIN:$ORIGIN/../lib"
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib"
)
install(TARGETS tutti_daemon DESTINATION "bin")
