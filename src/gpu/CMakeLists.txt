# ------------------------------------------------------------------------------
# engine_gpu: buffers, images, pipelines, swapchain, RenderDoc.
# ------------------------------------------------------------------------------

add_library(
    engine_gpu
    STATIC
        buffer.cxx
        context.cxx
        gpu_error_describe.cxx
        gpu_resource_table.cxx
        host_query_context.cxx
        image.cxx
        image_storage.cxx
        model_vertex.cxx
        queue_selection.cxx
        queue_set.cxx
        sampler_storage.cxx
        shader_binary_cache.cxx
        shader_object.cxx
        shader_object_storage.cxx
        submission_plan.cxx
        swapchain.cxx
        vk_barrier.cxx
        vma.cxx
)

target_include_directories(
    engine_gpu
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)

target_link_libraries(
    engine_gpu
    PUBLIC
        engine_deps
        engine_core
    PRIVATE
        engine_options
)

target_precompile_headers(
    engine_gpu
    REUSE_FROM
        engine_core
)

# ------------------------------------------------------------------------------
# VMA warning suppression. set_source_files_properties() is directory-scoped,
# so this has to live next to vma.cxx.
# ------------------------------------------------------------------------------

if(MSVC)
  set_source_files_properties(
      vma.cxx
      PROPERTIES
          COMPILE_OPTIONS "/W0"
  )
elseif(
    CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
    OR CMAKE_CXX_COMPILER_ID MATCHES "Clang"
)
  set_source_files_properties(
      vma.cxx
      PROPERTIES
          COMPILE_OPTIONS "-w"
  )
endif()

# ------------------------------------------------------------------------------
# RenderDoc: attaches renderdoc.cxx or stub.cxx.
# ------------------------------------------------------------------------------

configure_renderdoc(
    engine_gpu
    REAL_SOURCE
        "${CMAKE_CURRENT_SOURCE_DIR}/renderdoc/renderdoc.cxx"
    STUB_SOURCE
        "${CMAKE_CURRENT_SOURCE_DIR}/renderdoc/stub.cxx"
    INCLUDE_CACHE_VARIABLE
        RENDERDOC_INCLUDE_PATH
)
