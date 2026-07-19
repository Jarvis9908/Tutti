# io_engine/ -- Tutti IO Engine.
#
# Ships:
#   - tutti_io_engine_local_nvme    : pure host-side helper library
#                                     (NvmeBatchEntry / build_nvme_batch)
#                                     + the __global__ batch IO kernel
#                                     + launch_nvme_batch_xfer host wrapper.
#   - io_engine_host_smoke          : builds a batch + field-by-field
#                                     validates against a reference formula.
#   - io_engine_smoke               : end-to-end byte-verified roundtrip
#                                     (8 entries x 2-stripe GpuFile).
#
# Layer dependency:
#   io_engine -> memory          (IoSliceView, AddressDescriptor)
#             -> block_storage   (GpuFileHandle::d_shards_dev)
#             -> nvme_storage    (NvmeFileDeviceHandle, submit_*_one
#                                  via nvme_storage_device.cuh)

set(TUTTI_IO_ENGINE_LOCAL_NVME_SOURCES
    src/local_nvme/host_batch_builder.cpp
    src/local_nvme/local_nvme_io_engine.cpp     # IIoEngine impl
    src/local_nvme/nvme_batch_xfer_kernel.cu     # GPU kernel
)

add_library(tutti_io_engine_local_nvme ${TUTTI_IO_ENGINE_LOCAL_NVME_SOURCES})

target_include_directories(tutti_io_engine_local_nvme
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/local_nvme"
)

target_link_libraries(tutti_io_engine_local_nvme
    PUBLIC
        tutti_memory                # IMemorySubsystem, IoSliceView,
                                    # AddressDescriptor
        tutti_block_storage         # GpuFileHandle (d_shards_dev),
                                    # transitively nvme_storage's
                                    # NvmeFileDeviceHandle fwd-decl.
    PRIVATE
        libnvm                      # submit_*_one transitively
                                    # pulls libnvm primitives.
        CUDA::cudart                # cudaStream_t / launch.
)

set_target_properties(tutti_io_engine_local_nvme PROPERTIES
    OUTPUT_NAME              "tutti_io_engine_local_nvme"
    BUILD_RPATH              "$ORIGIN"
    INSTALL_RPATH            "$ORIGIN"
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON       # kernel calls __device__
    CUDA_RESOLVE_DEVICE_SYMBOLS ON      # submit_*_one in the libnvm-backed lib.
)

# ---------------------------------------------------------------------------
# Smoke tests live in the top-level examples/ tree.
# ---------------------------------------------------------------------------

install(TARGETS tutti_io_engine_local_nvme  DESTINATION "lib")
install(DIRECTORY include/ DESTINATION "include/tutti/io_engine"
        FILES_MATCHING PATTERN "*.h")
