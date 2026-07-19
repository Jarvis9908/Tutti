# adapters/kv_cache/ -- KV-cache semantics adapter (framework-facing).
#
# Implements the per-layer K/V byte-offset formula and auto-chunking
# on top of the generic Tutti io_engine, so the core runtime stays
# framework-agnostic (Roadmap.md "Adapter Layer").
#
# Pure host C++ (no CUDA device code): it only builds
# NvmeBatchInputTensor lists and calls Coordinator::submit_batch.
# cudaStream_t is used as an opaque handle, so a plain CXX TU suffices.
#
# Layer dependency:
#   tutti_kv_adapter -> tutti_coordinator (submit_batch + memory())
#                       which transitively exposes memory /
#                       block_storage / io_engine public headers.

add_library(tutti_kv_adapter
    src/kv_cache_io_adapter.cpp
)

target_include_directories(tutti_kv_adapter
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(tutti_kv_adapter
    PUBLIC
        tutti_coordinator       # Coordinator + transitive memory/block/io_engine
    PRIVATE
        CUDA::cudart
)

set_target_properties(tutti_kv_adapter PROPERTIES
    OUTPUT_NAME               "tutti_kv_adapter"
    BUILD_RPATH               "$ORIGIN"
    INSTALL_RPATH             "$ORIGIN"
    POSITION_INDEPENDENT_CODE ON
)

install(TARGETS tutti_kv_adapter DESTINATION "lib")
install(DIRECTORY include/ DESTINATION "include/tutti/adapters/kv_cache"
        FILES_MATCHING PATTERN "*.h")
