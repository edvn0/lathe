cmake_minimum_required(VERSION 3.25)

project(
    mingw-vulkan
    VERSION 0.1.0
    LANGUAGES C CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
    GIT_TAG master
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
    GIT_TAG master
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

add_library(BS_thread_pool INTERFACE)

target_include_directories(
    BS_thread_pool
    INTERFACE
        "${BS_thread_pool_SOURCE_DIR}/include"
)

CPMAddPackage(
    NAME bullet3
    GITHUB_REPOSITORY bulletphysics/bullet3
    GIT_TAG master
    GIT_SHALLOW YES
    OPTIONS
        "BT_THREADSAFE OFF"
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
# Project sources
# ------------------------------------------------------------------------------

file(
    GLOB_RECURSE mingw_vulkan_core_sources
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx"
)

list(
    FILTER mingw_vulkan_core_sources
    EXCLUDE REGEX "/platform/"
)

list(
    FILTER mingw_vulkan_core_sources
    EXCLUDE REGEX "/renderdoc/"
)

set(
    mingw_vulkan_main_source
    "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cxx"
)

if(NOT EXISTS "${mingw_vulkan_main_source}")
  message(
      FATAL_ERROR
      "Expected application entry point at "
      "${mingw_vulkan_main_source}"
  )
endif()

list(
    REMOVE_ITEM
    mingw_vulkan_core_sources
    "${mingw_vulkan_main_source}"
)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  file(
      GLOB_RECURSE platform_sources
      CONFIGURE_DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/src/platform/windows/*.cxx"
  )
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  file(
      GLOB_RECURSE platform_sources
      CONFIGURE_DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/src/platform/linux/*.cxx"
  )
else()
  set(platform_sources "")
endif()

list(
    APPEND
    mingw_vulkan_core_sources
    ${platform_sources}
)

file(
    GLOB_RECURSE mingw_vulkan_headers
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hxx"
)

# ------------------------------------------------------------------------------
# Core library
# ------------------------------------------------------------------------------

add_library(
    mingw-vulkan-core
    STATIC
        ${mingw_vulkan_core_sources}
        ${mingw_vulkan_headers}
)

add_library(
    mingw-vulkan::core
    ALIAS
    mingw-vulkan-core
)

target_include_directories(
    mingw-vulkan-core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
        "${stb_SOURCE_DIR}"
)

# Bullet's own CMakeLists uses directory-scoped INCLUDE_DIRECTORIES() rather
# than target_include_directories(), so its include path never reaches
# downstream targets via target_link_libraries() -- add it explicitly.
target_include_directories(
    mingw-vulkan-core
    SYSTEM
    PRIVATE
        "${bullet3_SOURCE_DIR}/src"
)

#
# These are intentionally PUBLIC for the initial core-library refactor.
#
# The old executable target received this entire dependency environment
# directly. Making it PUBLIC here preserves that behaviour for consumers of
# mingw-vulkan-core, including mingw-vulkan itself and the test targets.
#
# Individual libraries can later be tightened to PRIVATE where no public
# project header exposes them.
#
target_link_libraries(
    mingw-vulkan-core
    PUBLIC
        Vulkan::Headers
        volk::volk
        glfw
        BulletDynamics
        BulletCollision
        LinearMath
        spdlog::spdlog
        GPUOpen::VulkanMemoryAllocator
        fastgltf::fastgltf
        glm::glm
        EnTT::EnTT
        slang::headers
        mikktspace::mikktspace
        meshoptimizer
        efsw
        imgui
        BS_thread_pool
)

target_precompile_headers(
    mingw-vulkan-core
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include/pch.hxx"
)

target_compile_definitions(
    mingw-vulkan-core
    PUBLIC
        VK_NO_PROTOTYPES
        GLFW_INCLUDE_NONE
        GLM_FORCE_RADIANS
        GLM_FORCE_DEPTH_ZERO_TO_ONE
        GLM_ENABLE_EXPERIMENTAL
        SLANG_ROOT_PATH="${slang_root}"
)

if(MINGW_VULKAN_ENABLE_VALIDATION)
  target_compile_definitions(
      mingw-vulkan-core
      PUBLIC
          MINGW_VULKAN_ENABLE_VALIDATION=1
  )
else()
  target_compile_definitions(
      mingw-vulkan-core
      PUBLIC
          MINGW_VULKAN_ENABLE_VALIDATION=0
  )
endif()

# The error-context types (ErrorContext and the *Error aggregates' `.cause`/
# `.context` fields) are deliberately optional trailing members with defaults
# -- callers only ever fill in what's relevant to that failure. GCC's
# -Wmissing-field-initializers fires on every designated-init that omits them
# even though a default member initializer exists, which isn't a real bug.
target_compile_options(
    mingw-vulkan-core
    PRIVATE
        -Wno-missing-field-initializers
        -Wno-old-style-cast
)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/compiler_options.cmake")

configure_compiler_options(mingw-vulkan-core)

# ------------------------------------------------------------------------------
# RenderDoc
#
# RenderDoc implementation belongs to the core library, since the executable
# itself now contains only main.cxx.
# ------------------------------------------------------------------------------

include(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/renderdoc_config.cmake"
)

configure_renderdoc(
    mingw-vulkan-core
    REAL_SOURCE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/renderdoc/renderdoc.cxx"
    STUB_SOURCE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/renderdoc/stub.cxx"
    INCLUDE_CACHE_VARIABLE
        RENDERDOC_INCLUDE_PATH
)

# ------------------------------------------------------------------------------
# Application
# ------------------------------------------------------------------------------

add_executable(
    mingw-vulkan
    "${mingw_vulkan_main_source}"
)

target_link_libraries(
    mingw-vulkan
    PRIVATE
        mingw-vulkan::core
)

#
# Keep the same project compiler options on main.cxx as well. The majority of
# the source tree is compiled by mingw-vulkan-core, but the executable should
# not silently use a different warning/exception/RTTI configuration.
#
configure_compiler_options(mingw-vulkan)

# If main.cxx relies on the project PCH, retain the same behaviour it had when
# all sources belonged to the executable target.
target_precompile_headers(
    mingw-vulkan
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include/pch.hxx"
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
# VMA warning suppression
# ------------------------------------------------------------------------------

set(
    vma_source
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vma.cxx"
)

if(EXISTS "${vma_source}")
  if(MSVC)
    set_source_files_properties(
        "${vma_source}"
        PROPERTIES
            COMPILE_OPTIONS "/W0"
    )
  elseif(
      CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
      OR CMAKE_CXX_COMPILER_ID MATCHES "Clang"
  )
    set_source_files_properties(
        "${vma_source}"
        PROPERTIES
            COMPILE_OPTIONS "-w"
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
# Output directories
# ------------------------------------------------------------------------------

set_target_properties(
    mingw-vulkan
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

set_target_properties(
    mingw-vulkan-core
    PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

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