cmake_minimum_required(VERSION 3.25)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
project(
    mingw-vulkan
    VERSION 0.1.0
    LANGUAGES C CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Global output directories -- set once here rather than per-target, so every
# one of the 8 engine_<module> static libraries (and the executable) lands in
# the same place without an explicit set_target_properties() call each.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

include(CTest)

set(
    BUILD_SHARED_LIBS
    OFF
    CACHE BOOL
    "Build dependencies as shared libraries"
    FORCE
)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CPM.cmake")

# ------------------------------------------------------------------------------
# Mark third-party include directories as system includes
# ------------------------------------------------------------------------------

function(mark_target_includes_system target_name)
  if(NOT TARGET "${target_name}")
    return()
  endif()

  get_target_property(
        aliased_target
        "${target_name}"
        ALIASED_TARGET
  )

  if(aliased_target)
    set(real_target "${aliased_target}")
  else()
    set(real_target "${target_name}")
  endif()

  get_target_property(
        target_include_dirs
        "${real_target}"
        INTERFACE_INCLUDE_DIRECTORIES
  )

  if(
        NOT target_include_dirs
        OR target_include_dirs MATCHES "-NOTFOUND$"
  )
    return()
  endif()

  set_property(
        TARGET "${real_target}"
        APPEND
        PROPERTY
            INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
            "${target_include_dirs}"
  )
endfunction()

# ------------------------------------------------------------------------------
# Project options
# ------------------------------------------------------------------------------

option(
    MINGW_VULKAN_ENABLE_VALIDATION
    "Enable Vulkan validation support"
    ON
)

option(
    MINGW_VULKAN_ENABLE_EXCEPTIONS
    "Enable C++ exceptions"
    OFF
)

option(
    MINGW_VULKAN_ENABLE_TRACY
    "Enable Tracy profiler instrumentation"
    ON
)

option(
    MINGW_VULKAN_SANITIZE
    "Build with AddressSanitizer + UndefinedBehaviorSanitizer (native builds only)"
    OFF
)

option(
    MINGW_VULKAN_WERROR
    "Treat warnings as errors on this project's own targets (mingw-vulkan-core/mingw-vulkan/mingw-vulkan-tests)"
    OFF
)

set(
    SLANG_VERSION
    "2026.12.0.1"
    CACHE STRING
    "Slang release version"
)

set(
    SLANG_ROOT
    ""
    CACHE PATH
    "Path to an extracted Slang installation"
)

set(
    SLANG_ARCHIVE
    ""
    CACHE FILEPATH
    "Path to a local Slang release archive"
)

# engine_options only reads the MINGW_VULKAN_* options above; the shared
# third-party dependency target (engine_deps) is created later, once every
# CPMAddPackage() it references has run (see the "Core library" section).
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/compiler_options.cmake")

# ------------------------------------------------------------------------------
# Language configuration
# ------------------------------------------------------------------------------

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------

if(BUILD_TESTING AND NOT CMAKE_CROSSCOMPILING)
  CPMAddPackage(
      NAME doctest
      GITHUB_REPOSITORY doctest/doctest
      VERSION 2.5.3
  )
endif()

cpmaddpackage(
    NAME VulkanHeaders
    GITHUB_REPOSITORY KhronosGroup/Vulkan-Headers
    GIT_TAG vulkan-sdk-1.4.335.0
    OPTIONS
        "VULKAN_HEADERS_ENABLE_INSTALL OFF"
)

cpmaddpackage(
    NAME volk
    GITHUB_REPOSITORY zeux/volk
    GIT_TAG vulkan-sdk-1.4.335
)

cpmaddpackage(
    NAME glfw
    GITHUB_REPOSITORY glfw/glfw
    GIT_TAG 3.4
    OPTIONS
        "GLFW_BUILD_EXAMPLES OFF"
        "GLFW_BUILD_TESTS OFF"
        "GLFW_BUILD_DOCS OFF"
        "GLFW_INSTALL OFF"
)

cpmaddpackage(
    NAME spdlog
    GITHUB_REPOSITORY gabime/spdlog
    GIT_TAG v1.17.0
    OPTIONS
        "SPDLOG_BUILD_SHARED OFF"
        "SPDLOG_BUILD_EXAMPLE OFF"
        "SPDLOG_BUILD_EXAMPLE_HO OFF"
        "SPDLOG_BUILD_TESTS OFF"
        "SPDLOG_BUILD_TESTS_HO OFF"
        "SPDLOG_BUILD_BENCH OFF"
        "SPDLOG_INSTALL OFF"
        "SPDLOG_FMT_EXTERNAL OFF"
        "SPDLOG_FMT_EXTERNAL_HO OFF"
        "SPDLOG_USE_STD_FORMAT ON"
        "SPDLOG_NO_EXCEPTIONS ON"
)

cpmaddpackage(
    NAME VulkanMemoryAllocator
    GITHUB_REPOSITORY GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
    GIT_TAG v3.4.0
    OPTIONS
        "VMA_BUILD_DOCUMENTATION OFF"
        "VMA_BUILD_SAMPLES OFF"
)

cpmaddpackage(
    NAME fastgltf
    GITHUB_REPOSITORY spnda/fastgltf
    GIT_TAG v0.9.0
    OPTIONS
        "FASTGLTF_ENABLE_DEPRECATED_EXT OFF"
        "FASTGLTF_ENABLE_TESTS OFF"
        "FASTGLTF_ENABLE_EXAMPLES OFF"
        "FASTGLTF_ENABLE_DOCS OFF"
)

cpmaddpackage(
    NAME glm
    GITHUB_REPOSITORY g-truc/glm
    GIT_TAG 1.0.3
    SYSTEM YES
    OPTIONS
        "GLM_BUILD_TESTS OFF"
        "GLM_BUILD_INSTALL OFF"
)

cpmaddpackage(
    NAME EnTT
    GITHUB_REPOSITORY skypjack/entt
    GIT_TAG v3.15.0
)

cpmaddpackage(
    NAME meshoptimizer
    GITHUB_REPOSITORY zeux/meshoptimizer
    GIT_TAG v0.24
    OPTIONS
        "MESHOPT_BUILD_DEMO OFF"
        "MESHOPT_BUILD_TOOLS OFF"
        "MESHOPT_BUILD_SHARED_LIBS OFF"
        "MESHOPT_BUILD_WERROR OFF"
)

cpmaddpackage(
    NAME stb
    GITHUB_REPOSITORY nothings/stb
    GIT_TAG 2c980bb59875b0d32144a71867fbdebb2f77cd20
    GIT_SHALLOW YES
    DOWNLOAD_ONLY YES
)

CPMAddPackage(
    NAME efsw
    GITHUB_REPOSITORY SpartanJ/efsw
    GIT_TAG 1.4.1
    OPTIONS
        "BUILD_SHARED_LIBS OFF"
)

set(USE_CCACHE YES)

CPMAddPackage(
    "gh:TheLartians/Ccache.cmake@1.2.5"
)

CPMAddPackage(
    NAME freetype
    GITHUB_REPOSITORY freetype/freetype
    GIT_TAG VER-2-14-3
    OPTIONS
        "BUILD_SHARED_LIBS OFF"
        "FT_DISABLE_ZLIB ON"
        "FT_DISABLE_BZIP2 ON"
        "FT_DISABLE_PNG ON"
        "FT_DISABLE_HARFBUZZ ON"
        "FT_DISABLE_BROTLI ON"
)

CPMAddPackage(
    NAME BS_thread_pool
    GITHUB_REPOSITORY bshoshany/thread-pool
    VERSION 5.1.0
    EXCLUDE_FROM_ALL
    SYSTEM
)

# TRACY_ON_DEMAND defers data collection until a profiler actually connects,
# so an unattached run doesn't buffer an ever-growing history in memory.
CPMAddPackage(
    NAME tracy
    GITHUB_REPOSITORY wolfpld/tracy
    GIT_TAG v0.14.1
    OPTIONS
        "TRACY_ENABLE ${MINGW_VULKAN_ENABLE_TRACY}"
        "TRACY_ON_DEMAND ON"
)

add_library(BS_thread_pool INTERFACE)

target_include_directories(
    BS_thread_pool
    INTERFACE
        "${BS_thread_pool_SOURCE_DIR}/include"
)

CPMAddPackage(
    NAME bullet3
    GITHUB_REPOSITORY bulletphysics/bullet3
    GIT_TAG 3.25
    GIT_SHALLOW YES
    OPTIONS
        # This project uses Bullet's *Mt classes (btCollisionDispatcherMt,
        # btConstraintSolverPoolMt, btDiscreteDynamicsWorldMt) and installs a
        # custom btITaskScheduler (see physics_world.cxx), which requires
        # Bullet's own mutex primitives to be compiled in. Bullet gates that
        # behind BULLET2_MULTITHREADING (defaults OFF), not a "BT_THREADSAFE"
        # cache variable -- that name isn't read anywhere in Bullet's
        # CMakeLists, so setting it here was a no-op. Without the real
        # option, LinearMath/btThreads.cpp compiles btSpinMutex::lock() as an
        # assert-and-abort stub, and every *Mt class call that needs it
        # crashes the moment Bullet's internal dispatch is busy enough to
        # actually take that lock -- confirmed by test/physics_world_test.cxx
        # aborting with "unimplemented btSpinMutex::lock() called" before
        # this option was corrected.
        "BULLET2_MULTITHREADING ON"
        "BUILD_SHARED_LIBS OFF"
        "BUILD_BULLET2_DEMOS OFF"
        "BUILD_CPU_DEMOS OFF"
        "BUILD_OPENGL3_DEMOS OFF"
        "BUILD_EXTRAS OFF"
        "BUILD_UNIT_TESTS OFF"
        "BUILD_PYBULLET OFF"
        "USE_DOUBLE_PRECISION OFF"
        "USE_GRAPHICAL_BENCHMARK OFF"
        "INSTALL_LIBS OFF"
)

if(bullet3_SOURCE_DIR)
  # Bullet is old code that never sets its own C++ standard, and the
  # directory-scoped CMAKE_CXX_STANDARD 23 / CMAKE_CXX_EXTENSIONS OFF above
  # would otherwise leak into it -- pin it back to a standard it's actually
  # written against.
  foreach(bullet_target IN ITEMS BulletDynamics BulletCollision LinearMath)
    if(TARGET ${bullet_target})
      set_target_properties(
          ${bullet_target}
          PROPERTIES
              CXX_STANDARD 17
              CXX_STANDARD_REQUIRED ON
              CXX_EXTENSIONS OFF
      )
    endif()
  endforeach()
endif()

CPMAddPackage(
    NAME ImGui
    GITHUB_REPOSITORY ocornut/imgui
    GIT_TAG v1.92.5-docking
    GIT_SHALLOW YES
    DOWNLOAD_ONLY YES
)

CPMAddPackage(
    NAME ImPlot
    GITHUB_REPOSITORY epezent/implot
    GIT_TAG bd99f8d8b9a80c493782ab8cb3335af102f61304
    GIT_SHALLOW NO
    DOWNLOAD_ONLY YES
)

CPMAddPackage(
    NAME ImGuizmo
    GITHUB_REPOSITORY CedricGuillemet/ImGuizmo
    GIT_TAG a15acd87a3f3241a29ea1363ceafc680dca3a96b
    GIT_SHALLOW NO
    DOWNLOAD_ONLY YES
)

CPMAddPackage(
    NAME tinyexr
    GITHUB_REPOSITORY syoyo/tinyexr
    GIT_TAG v3.2.0
    DOWNLOAD_ONLY YES
)

file(
    GLOB TINYEXR_SOURCES
    CONFIGURE_DEPENDS
    "${tinyexr_SOURCE_DIR}/src/*.c"
)

add_library(
    tinyexr
    STATIC
    ${TINYEXR_SOURCES}
    "${tinyexr_SOURCE_DIR}/deps/zstd/tinyexr_zstd.c"
)

target_include_directories(
    tinyexr
    PUBLIC
        "${tinyexr_SOURCE_DIR}/include"
    PRIVATE
        "${tinyexr_SOURCE_DIR}/src"
        "${tinyexr_SOURCE_DIR}/deps/zstd"
)

set_target_properties(
    tinyexr
    PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED YES
        C_EXTENSIONS NO
)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tinyexr PRIVATE -w)
endif()

# ------------------------------------------------------------------------------
# KTX-Software (libktx): KTX2 container read/write plus the Basis Universal
# UASTC encoder/transcoder used to bake BC5/BC7 textures. Tools, loadtests,
# and its own GL/Vulkan upload paths are all disabled -- we only need the
# library, and we do our own Vulkan uploads.
# ------------------------------------------------------------------------------

set(KTX_FEATURE_TESTS OFF CACHE BOOL "" FORCE)
set(KTX_FEATURE_TOOLS OFF CACHE BOOL "" FORCE)
set(KTX_FEATURE_TOOLS_CTS OFF CACHE BOOL "" FORCE)
set(KTX_FEATURE_DOC OFF CACHE BOOL "" FORCE)
set(KTX_FEATURE_LOADTEST_APPS OFF CACHE STRING "" FORCE)
set(KTX_FEATURE_JNI OFF CACHE BOOL "" FORCE)
set(KTX_FEATURE_PY OFF CACHE BOOL "" FORCE)
set(KTX_FEATURE_GL_UPLOAD OFF CACHE BOOL "" FORCE)
set(KTX_FEATURE_VK_UPLOAD OFF CACHE BOOL "" FORCE)

# KTX-Software vendors large LFS-tracked test/reference images that are
# irrelevant with KTX_FEATURE_TESTS/TOOLS_CTS off; skip fetching their real
# contents (LFS pointer files are fine -- nothing in the ktx target reads
# them), and skip the tests/cts submodule entirely, since it exists only for
# KTX_FEATURE_TOOLS_CTS.
set(ENV{GIT_LFS_SKIP_SMUDGE} "1")

CPMAddPackage(
    NAME ktx
    GITHUB_REPOSITORY KhronosGroup/KTX-Software
    GIT_TAG v4.4.1
    GIT_SUBMODULES ""
)

mark_target_includes_system(ktx)

if(ImGui_SOURCE_DIR AND ImPlot_SOURCE_DIR AND ImGuizmo_SOURCE_DIR)
  add_library(
      imgui
      STATIC
          ${ImGui_SOURCE_DIR}/imgui.cpp
          ${ImGui_SOURCE_DIR}/imgui_demo.cpp
          ${ImGui_SOURCE_DIR}/imgui_draw.cpp
          ${ImGui_SOURCE_DIR}/imgui_tables.cpp
          ${ImGui_SOURCE_DIR}/imgui_widgets.cpp
          ${ImGui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
          ${ImGui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
          ${ImGui_SOURCE_DIR}/misc/freetype/imgui_freetype.cpp
          ${ImPlot_SOURCE_DIR}/implot.cpp
          ${ImPlot_SOURCE_DIR}/implot_items.cpp
          ${ImGuizmo_SOURCE_DIR}/ImGuizmo.cpp
  )

  target_include_directories(
      imgui
      PUBLIC
          ${ImGui_SOURCE_DIR}
          ${ImGui_SOURCE_DIR}/backends
          ${ImPlot_SOURCE_DIR}
          ${ImGuizmo_SOURCE_DIR}
  )

  target_link_libraries(
      imgui
      PUBLIC
          glfw
          volk
          volk::volk_headers
          freetype
  )

  target_compile_definitions(
      imgui
      PUBLIC
          GLFW_INCLUDE_NONE
          IMGUI_IMPL_VULKAN_USE_VOLK
          IMGUI_ENABLE_FREETYPE
  )

  find_package(X11 REQUIRED)

  if(UNIX AND NOT APPLE)
    target_link_libraries(
        imgui
        PUBLIC
            X11::X11
    )
  endif()
endif()

mark_target_includes_system(meshoptimizer)
mark_target_includes_system(efsw)
mark_target_includes_system(TracyClient)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/add_mikktspace.cmake")

add_mikktspace_dependency()

mark_target_includes_system(mikktspace::mikktspace)

# ------------------------------------------------------------------------------
# Slang binary distribution
#
# Resolution order:
#
#   1. SLANG_ROOT
#   2. SLANG_ARCHIVE
#   3. CPM network download
# ------------------------------------------------------------------------------

function(validate_slang_root root_path)
  if(NOT IS_DIRECTORY "${root_path}")
    message(
        FATAL_ERROR
        "Slang root is not a directory: ${root_path}"
    )
  endif()

  if(NOT EXISTS "${root_path}/include/slang.h")
    message(
        FATAL_ERROR
        "Slang installation does not contain include/slang.h: "
        "${root_path}"
    )
  endif()

  if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    if(NOT EXISTS "${root_path}/bin/slang.dll")
      message(
          FATAL_ERROR
          "Windows Slang installation does not contain bin/slang.dll: "
          "${root_path}"
      )
    endif()
  elseif(APPLE)
    if(
        NOT EXISTS "${root_path}/lib/libslang.dylib"
        AND NOT EXISTS "${root_path}/bin/libslang.dylib"
    )
      message(
          FATAL_ERROR
          "macOS Slang installation does not contain libslang.dylib: "
          "${root_path}"
      )
    endif()
  else()
    if(
        NOT EXISTS "${root_path}/lib/libslang.so"
        AND NOT EXISTS "${root_path}/bin/libslang.so"
    )
      message(
          FATAL_ERROR
          "Linux Slang installation does not contain libslang.so: "
          "${root_path}"
      )
    endif()
  endif()
endfunction()

set(slang_root "")

if(SLANG_ROOT)
  cmake_path(
      ABSOLUTE_PATH SLANG_ROOT
      BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      NORMALIZE
      OUTPUT_VARIABLE slang_root
  )

  message(
      STATUS
      "Using extracted local Slang installation: ${slang_root}"
  )

elseif(SLANG_ARCHIVE)
  cmake_path(
      ABSOLUTE_PATH SLANG_ARCHIVE
      BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      NORMALIZE
      OUTPUT_VARIABLE slang_archive
  )

  if(NOT EXISTS "${slang_archive}")
    message(
        FATAL_ERROR
        "SLANG_ARCHIVE does not exist: ${slang_archive}"
    )
  endif()

  message(
      STATUS
      "Using local Slang archive: ${slang_archive}"
  )

  cpmaddpackage(
      NAME slang_binary
      URL "${slang_archive}"
      DOWNLOAD_ONLY YES
  )

  set(slang_root "${slang_binary_SOURCE_DIR}")

else()
  if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(
        slang_archive_name
        "slang-${SLANG_VERSION}-windows-x86_64.tar.gz"
    )
  elseif(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|ARM64)$")
      set(
          slang_archive_name
          "slang-${SLANG_VERSION}-macos-aarch64.tar.gz"
      )
    else()
      set(
          slang_archive_name
          "slang-${SLANG_VERSION}-macos-x86_64.tar.gz"
      )
    endif()
  else()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|ARM64)$")
      set(
          slang_archive_name
          "slang-${SLANG_VERSION}-linux-aarch64.tar.gz"
      )
    else()
      set(
          slang_archive_name
          "slang-${SLANG_VERSION}-linux-x86_64.tar.gz"
      )
    endif()
  endif()

  string(
      CONCAT
      slang_download_url
      "https://github.com/shader-slang/slang/releases/download/"
      "v${SLANG_VERSION}/"
      "${slang_archive_name}"
  )

  message(
      STATUS
      "Downloading Slang ${SLANG_VERSION}: ${slang_archive_name}"
  )

  cpmaddpackage(
      NAME slang_binary
      URL "${slang_download_url}"
      DOWNLOAD_ONLY YES
  )

  set(slang_root "${slang_binary_SOURCE_DIR}")
endif()

validate_slang_root("${slang_root}")

set(slang_include_dir "${slang_root}/include")
set(slang_bin_dir "${slang_root}/bin")
set(slang_lib_dir "${slang_root}/lib")

add_library(slang_binary_headers INTERFACE)
add_library(slang::headers ALIAS slang_binary_headers)

target_include_directories(
    slang_binary_headers
    SYSTEM
    INTERFACE
        "${slang_include_dir}"
)

mark_target_includes_system(Vulkan::Headers)
mark_target_includes_system(volk::volk)
mark_target_includes_system(glfw)
mark_target_includes_system(spdlog::spdlog)
mark_target_includes_system(GPUOpen::VulkanMemoryAllocator)
mark_target_includes_system(fastgltf::fastgltf)
mark_target_includes_system(glm::glm)
mark_target_includes_system(EnTT::EnTT)

# slang_binary_headers already uses SYSTEM above.

# ------------------------------------------------------------------------------
# Shader push-constant reflection
# ------------------------------------------------------------------------------

CPMAddPackage(
    NAME SPIRV-Reflect
    GITHUB_REPOSITORY KhronosGroup/SPIRV-Reflect
    GIT_TAG vulkan-sdk-1.4.335.0
    DOWNLOAD_ONLY YES
)

# The reflection tool is always a host executable.
find_program(
    shader_reflect_host_compiler
    NAMES g++ c++ clang++
    REQUIRED
)

# slangc must likewise execute on the host.
if(CMAKE_CROSSCOMPILING)
    if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|ARM64)$")
        set(
            shader_reflect_host_slang_archive_name
            "slang-${SLANG_VERSION}-linux-aarch64.tar.gz"
        )
    else()
        set(
            shader_reflect_host_slang_archive_name
            "slang-${SLANG_VERSION}-linux-x86_64.tar.gz"
        )
    endif()

    string(
        CONCAT
        shader_reflect_host_slang_url
        "https://github.com/shader-slang/slang/releases/download/"
        "v${SLANG_VERSION}/"
        "${shader_reflect_host_slang_archive_name}"
    )

    CPMAddPackage(
        NAME slang_binary_host
        URL "${shader_reflect_host_slang_url}"
        DOWNLOAD_ONLY YES
    )

    set(
        shader_reflect_slangc
        "${slang_binary_host_SOURCE_DIR}/bin/slangc"
    )
else()
    set(
        shader_reflect_slangc
        "${slang_bin_dir}/slangc"
    )
endif()

set(
    shader_reflect_generated_dir
    "${CMAKE_BINARY_DIR}/generated"
)

set(
    shader_reflect_tool
    "${shader_reflect_generated_dir}/tools/reflect_push_constants"
)

set(
    shader_reflect_tool_thread_flags
)

if(CMAKE_HOST_SYSTEM_NAME MATCHES "Linux|Darwin")
    list(
        APPEND
        shader_reflect_tool_thread_flags
        -pthread
    )
endif()

add_custom_command(
    OUTPUT
        "${shader_reflect_tool}"

    COMMAND
        "${CMAKE_COMMAND}"
        -E
        make_directory
        "${shader_reflect_generated_dir}/tools"

    COMMAND
        "${shader_reflect_host_compiler}"
        -std=c++23
        -O2
        ${shader_reflect_tool_thread_flags}
        "-I${SPIRV-Reflect_SOURCE_DIR}"
        "-I${BS_thread_pool_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/shader_reflect/reflect_push_constants.cxx"
        "${SPIRV-Reflect_SOURCE_DIR}/spirv_reflect.c"
        -o
        "${shader_reflect_tool}"

    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/shader_reflect/reflect_push_constants.cxx"
        "${SPIRV-Reflect_SOURCE_DIR}/spirv_reflect.c"
        "${SPIRV-Reflect_SOURCE_DIR}/spirv_reflect.h"

    VERBATIM
)

set(
    shader_reflect_spv_outputs
)

set(
    shader_reflect_arguments
)

#
# add_shader_push_constant(
#     <slang file>
#     <entry point>
#     <stage>
#     <generated struct name>
# )
#
# Each shader is compiled independently, allowing Ninja to parallelise Slang
# compilation. The resulting SPIR-V files are reflected together in one host
# process below.
#
macro(
    add_shader_push_constant
    slang_file
    entry_point
    stage
    struct_name
)
    set(
        shader_source
        "${CMAKE_CURRENT_SOURCE_DIR}/assets/shaders/${slang_file}"
    )

    set(
        spv_path
        "${shader_reflect_generated_dir}/spv/${struct_name}.spv"
    )

    set(
        depfile_path
        "${shader_reflect_generated_dir}/spv/${struct_name}.d"
    )

    add_custom_command(
        OUTPUT
            "${spv_path}"

        COMMAND
            "${CMAKE_COMMAND}"
            -E
            make_directory
            "${shader_reflect_generated_dir}/spv"

        COMMAND
            "${shader_reflect_slangc}"
            "${shader_source}"
            -entry
            "${entry_point}"
            -stage
            "${stage}"
            -target
            spirv
            -profile
            spirv_1_6
            -matrix-layout-column-major
            -force-glsl-scalar-layout
            -I
            "${CMAKE_CURRENT_SOURCE_DIR}/assets/shaders"
            -depfile
            "${depfile_path}"
            -o
            "${spv_path}"

        DEPENDS
            "${shader_source}"

        DEPFILE
            "${depfile_path}"

        VERBATIM
    )

    list(
        APPEND
        shader_reflect_spv_outputs
        "${spv_path}"
    )

    list(
        APPEND
        shader_reflect_arguments
        --shader
        "${struct_name}"
        "${spv_path}"
    )
endmacro()

add_shader_push_constant(
    forward_geom.slang
    mainVs
    vertex
    ForwardPushConstants
)

add_shader_push_constant(
    shadow_depth.slang
    mainVs
    vertex
    ShadowPushConstants
)

add_shader_push_constant(
    composite.slang
    mainFs
    fragment
    CompositePushConstants
)

add_shader_push_constant(
    light_icons.slang
    main_task
    amplification
    LightIconPushConstants
)

add_shader_push_constant(
    frustum_cull.slang
    mainCs
    compute
    CullPushConstants
)

add_shader_push_constant(
    bloom_downsample.slang
    mainCs
    compute
    DownsamplePushConstants
)

add_shader_push_constant(
    bloom_upsample.slang
    mainCs
    compute
    UpsamplePushConstants
)

add_shader_push_constant(
    gtao.slang
    mainCs
    compute
    GtaoPushConstants
)

add_shader_push_constant(
    gtao_denoise.slang
    mainCs
    compute
    GtaoDenoisePushConstants
)

set(
    shader_push_constants_header
    "${shader_reflect_generated_dir}/include/shader_push_constants.hxx"
)

set(
    shader_push_constants_preamble
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/shader_push_constants_preamble.hxx"
)

add_custom_command(
    OUTPUT
        "${shader_push_constants_header}"

    COMMAND
        "${shader_reflect_tool}"
        --output
        "${shader_push_constants_header}"
        --preamble
        "${shader_push_constants_preamble}"
        ${shader_reflect_arguments}

    DEPENDS
        "${shader_reflect_tool}"
        "${shader_push_constants_preamble}"
        ${shader_reflect_spv_outputs}

    COMMAND_EXPAND_LISTS
    VERBATIM
)

add_custom_target(
    generate_shader_push_constants
    DEPENDS
        "${shader_push_constants_header}"
)

# ------------------------------------------------------------------------------
# Project sources
#
# The 8 engine_<module> static libraries are configured by their own
# src/<module>/CMakeLists.txt (explicit source lists, not file(GLOB) --
# the point of the split is for each module's source list to document its
# own contents). This section only resolves what's shared by all of them,
# plus the sources that stay directly on the executable rather than any
# module library.
# ------------------------------------------------------------------------------

#
# The global operator new/delete override (global_new_delete.cxx) routes every
# allocation through MemoryTracker for the in-app Memory panel. That's a
# per-allocation cost worth paying in Debug/RelWithDebInfo, but not something
# a Release build should carry -- Release gets no tracking overhead and, on
# Linux, links jemalloc instead (see the "Release allocator" section below).
#
# A global new/delete override inside a static archive can be silently
# dropped by the linker if nothing else in that archive member is
# referenced, so this stays a source of the executable itself rather than
# any engine_<module> library.
#
set(mingw_vulkan_track_memory ON)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(mingw_vulkan_track_memory OFF)
endif()

#
# main.cxx is just the entry point/frame loop/window-event wiring;
# vulkan_bootstrap.cxx holds the one-time instance/device/swapchain bring-up
# it calls into. Neither belongs to any engine_<module> library, since
# nothing else links against them.
#
set(
    mingw_vulkan_main_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cxx"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan_bootstrap.cxx"
)

foreach(mingw_vulkan_main_source ${mingw_vulkan_main_sources})
  if(NOT EXISTS "${mingw_vulkan_main_source}")
    message(
        FATAL_ERROR
        "Expected application entry point source at "
        "${mingw_vulkan_main_source}"
    )
  endif()
endforeach()

if(mingw_vulkan_track_memory)
  list(
      APPEND mingw_vulkan_main_sources
      "${CMAKE_CURRENT_SOURCE_DIR}/src/global_new_delete.cxx"
  )
endif()

#
# Game sources -- linked only into the mingw-vulkan executable, never into
# any engine_<module> library, so the module libraries stay entirely
# game-agnostic (see include/app/game.hxx). game/include is added as a
# private include directory on the executable target below, separate from
# the module libraries' public include/.
#
file(
    GLOB_RECURSE mingw_vulkan_game_sources
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/game/src/*.cxx"
)

list(
    APPEND mingw_vulkan_main_sources
    ${mingw_vulkan_game_sources}
)

# engine_deps needs stb_SOURCE_DIR, bullet3_SOURCE_DIR, shader_reflect_generated_dir
# and slang_root, all resolved above -- must be included after them.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/engine_deps.cmake")

# configure_renderdoc(), called from src/gpu/CMakeLists.txt below, must be
# defined before that add_subdirectory() runs.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/renderdoc_config.cmake")

# ------------------------------------------------------------------------------
# Module libraries
#
# Added in dependency order (core < gpu < assets < physics < scene < terrain
# < rendering < app) for readability; CMake itself doesn't require this
# order since target_link_libraries() may reference a not-yet-defined
# target as long as it exists by generate time.
# ------------------------------------------------------------------------------

add_subdirectory(src/core)
add_subdirectory(src/gpu)
add_subdirectory(src/assets)
add_subdirectory(src/physics)
add_subdirectory(src/scene)
add_subdirectory(src/terrain)
add_subdirectory(src/rendering)
add_subdirectory(src/app)

# ------------------------------------------------------------------------------
# Core library
#
# mingw-vulkan-core is now a thin INTERFACE umbrella over the 8
# engine_<module> libraries, so downstream consumers (the executable,
# test/CMakeLists.txt) don't need to know the module list -- this alias
# already existed pre-split, this just repoints what it aliases.
#
# $<LINK_GROUP:RESCAN,...> wraps the 8 libraries the way -Wl,--start-group /
# --end-group would: a plain left-to-right static-archive link only ever
# rescans an *earlier* archive for a symbol a *later* one turns out to need,
# never the reverse, so a strictly-layered link order breaks the two
# deliberate exceptions to that layering that were fine as a single archive
# but aren't once it's 8 separate ones -- error_describe.cxx (core) defining
# the generic describe() dispatcher that calls per-subsystem overloads
# defined in gpu/assets/rendering (Phase 2 point 9), and renderer.cxx
# (rendering) instantiating Renderer::record_frame<ApplicationOverlayPolicy>,
# which calls back into renderer_application_policy.cxx (app) (Phase 3's
# "one deliberate header/impl module split"). RESCAN makes the linker keep
# revisiting the whole group until every symbol resolves, in any direction.
# ------------------------------------------------------------------------------

add_library(
    mingw-vulkan-core
    INTERFACE
)

target_link_libraries(
    mingw-vulkan-core
    INTERFACE
        "$<LINK_GROUP:RESCAN,engine_app,engine_rendering,engine_terrain,engine_scene,engine_physics,engine_assets,engine_gpu,engine_core>"
)

add_library(
    mingw-vulkan::core
    ALIAS
    mingw-vulkan-core
)

# ------------------------------------------------------------------------------
# Application
# ------------------------------------------------------------------------------

add_executable(
    mingw-vulkan
    ${mingw_vulkan_main_sources}
)

target_link_libraries(
    mingw-vulkan
    PRIVATE
        mingw-vulkan::core
        engine_options
)

target_include_directories(
    mingw-vulkan
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/game/include"
)

# If these sources rely on the project PCH, retain the same behaviour they had
# when all sources belonged to the executable target.
target_precompile_headers(
    mingw-vulkan
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include/core/pch.hxx"
)

# ------------------------------------------------------------------------------
# Platform configuration
#
# Keep final executable/link/runtime deployment configuration on the executable.
#
# Platform source files themselves are compiled into mingw-vulkan-core above.
# ------------------------------------------------------------------------------

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  include(
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/platform_windows.cmake"
  )

  configure_windows_target(
      mingw-vulkan
      "${slang_bin_dir}"
  )

elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  include(
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/platform_linux.cmake"
  )

  configure_linux_target(
      mingw-vulkan
      "${slang_lib_dir}"
      "${slang_bin_dir}"
  )

else()
  message(
      WARNING
      "No platform-specific target configuration for "
      "${CMAKE_SYSTEM_NAME}; Slang runtime will not be deployed "
      "automatically."
  )
endif()

# ------------------------------------------------------------------------------
# Release allocator (Linux only)
#
# windows-mingw Release just gets the plain system allocator -- jemalloc's
# Windows/mingw-w64 support isn't solid enough to depend on for a
# cross-compiled target. linux-native Release links jemalloc, which overrides
# malloc/free (and therefore the default operator new/delete, which call
# through to malloc) via ordinary symbol interposition -- no source changes
# needed on our side. Requires libjemalloc-dev in the build image; see
# Dockerfile.
#
# Attached to the executable rather than any engine_<module> library: like
# global_new_delete.cxx, its effect depends on final link-line position, not
# on which archive happens to reference it.
# ------------------------------------------------------------------------------

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_BUILD_TYPE STREQUAL "Release")
  find_library(MINGW_VULKAN_JEMALLOC_LIBRARY jemalloc)

  if(MINGW_VULKAN_JEMALLOC_LIBRARY)
    target_link_libraries(
        mingw-vulkan
        PRIVATE
            "${MINGW_VULKAN_JEMALLOC_LIBRARY}"
    )
  else()
    message(
        WARNING
        "libjemalloc not found -- linux-native Release will use the plain "
        "system allocator. Install libjemalloc-dev to enable it."
    )
  endif()
endif()

# ------------------------------------------------------------------------------
# Assets
# ------------------------------------------------------------------------------

set(
    project_asset_directories
    assets
)

foreach(asset_directory IN LISTS project_asset_directories)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${asset_directory}")
    add_custom_command(
        TARGET mingw-vulkan
        POST_BUILD
        COMMAND
            "${CMAKE_COMMAND}" -E copy_directory_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/${asset_directory}"
            "$<TARGET_FILE_DIR:mingw-vulkan>/${asset_directory}"
        VERBATIM
    )
  endif()
endforeach()

# ------------------------------------------------------------------------------
# Tests
#
# doctest_discover_tests executes the resulting test binary to enumerate
# TEST_CASEs. For now, only enable automatic discovery for native builds.
# ------------------------------------------------------------------------------

if(BUILD_TESTING AND NOT CMAKE_CROSSCOMPILING)
  include(
      "${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake"
  )

  add_subdirectory(test)
elseif(BUILD_TESTING AND CMAKE_CROSSCOMPILING)
  message(
      STATUS
      "Skipping native unit tests while cross-compiling"
  )
endif()

# ------------------------------------------------------------------------------
# Configuration summary
# ------------------------------------------------------------------------------

message(
    STATUS
    "Slang root: ${slang_root}"
)

message(
    STATUS
    "Slang include directory: ${slang_include_dir}"
)

message(
    STATUS
    "Slang binary directory: ${slang_bin_dir}"
)

message(
    STATUS
    "C++ exceptions enabled: ${MINGW_VULKAN_ENABLE_EXCEPTIONS}"
)

if(SLANG_ROOT)
  message(
      STATUS
      "Slang source: extracted local installation"
  )
elseif(SLANG_ARCHIVE)
  message(
      STATUS
      "Slang source: local archive"
  )
else()
  message(
      STATUS
      "Slang source: CPM download"
  )
endif()

