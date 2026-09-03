# ------------------------------------------------------------------------------
# engine_gpu -- Vulkan resource/pipeline layer (buffers, images, pipelines,
# swapchain, RenderDoc integration). Depends only on engine_core.
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
        pipeline.cxx
        pipeline_storage.cxx
        sampler_storage.cxx
        shader_object.cxx
        shader_object_storage.cxx
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
# VMA warning suppression
#
# set_source_files_properties() is directory-scoped in CMake -- this must live
# in vma.cxx's own CMakeLists.txt, not the root one, or it silently stops
# applying and VMA's (extensive) warnings return, failing the build under
# -Werror/WERROR=1.
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
# RenderDoc
#
# configure_renderdoc() (defined in cmake/renderdoc_config.cmake, included by
# the root CMakeLists.txt before any add_subdirectory() call) attaches exactly
# one of renderdoc.cxx/stub.cxx to engine_gpu, the module that owns
# renderdoc.hxx/.cxx.
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
