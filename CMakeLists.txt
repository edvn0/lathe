cmake_minimum_required(VERSION 3.25)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
project(
    lathe
    VERSION 0.1.0
    LANGUAGES C CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Shared output directories for every module library and the executable.
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
    LATHE_ENABLE_VALIDATION
    "Enable Vulkan validation support"
    ON
)

option(
    LATHE_ENABLE_EXCEPTIONS
    "Enable C++ exceptions"
    OFF
)

option(
    LATHE_ENABLE_TRACY
    "Enable Tracy profiler instrumentation"
    ON
)

option(
    LATHE_SANITIZE
    "Build with AddressSanitizer + UndefinedBehaviorSanitizer (native builds only)"
    OFF
)

option(
    LATHE_WERROR
    "Treat warnings as errors on this project's own targets (lathe-core/lathe/lathe-tests)"
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

# engine_deps is created later, once every package it references has been added.
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

# libcurl (HTTP client) and libcrypto (SHA-256) come from the system; the build image installs the dev packages.
find_package(CURL REQUIRED)
find_package(OpenSSL REQUIRED)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/spirv-tools-opt.cmake")
find_spirv_tools_opt()

if(BUILD_TESTING)
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

# TRACY_ON_DEMAND only collects data once a profiler connects, so unattached runs don't buffer history.
CPMAddPackage(
    NAME tracy
    GITHUB_REPOSITORY wolfpld/tracy
    GIT_TAG v0.14.1
    OPTIONS
        "TRACY_ENABLE ${LATHE_ENABLE_TRACY}"
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
        # The *Mt Bullet classes and our custom btITaskScheduler need Bullet's real mutexes, which are only compiled
        # with BULLET2_MULTITHREADING. Without it btSpinMutex::lock() aborts.
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
  # Bullet doesn't set its own C++ standard; keep it off our C++23 settings.
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

# Header-only toasts. Only src/rendering/toast.cxx includes it (see gui::toast_*).
CPMAddPackage(
    NAME ImGuiNotify
    GITHUB_REPOSITORY TyomaVader/ImGuiNotify
    GIT_TAG d00e45f8d6b1e094bc9288d20eb3d2840f6a7d73
    GIT_SHALLOW NO
    DOWNLOAD_ONLY YES
    # The dismiss button erases the toast and then keeps reading through the erased element.
    PATCHES "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/imgui_notify_dismiss_button.patch"
)

# ------------------------------------------------------------------------------
# Scripting: Lua 5.4 (compiled as C; errors are setjmp/longjmp) + sol2.
# ------------------------------------------------------------------------------

CPMAddPackage(
    NAME lua
    GITHUB_REPOSITORY lua/lua
    GIT_TAG v5.4.8
    DOWNLOAD_ONLY YES
)

# The core plus only the libraries the sandbox opens. io/os/package(loadlib)/debug/coroutine/utf8 and linit.c
# (which references all of them) are not in the binary at all. lua.c, onelua.c and ltests.c are front ends/tests.
set(
    lathe_lua_sources
    lapi.c lcode.c lctype.c ldebug.c ldo.c ldump.c lfunc.c lgc.c llex.c lmem.c lobject.c lopcodes.c
    lparser.c lstate.c lstring.c ltable.c ltm.c lundump.c lvm.c lzio.c
    lauxlib.c lbaselib.c lstrlib.c ltablib.c lmathlib.c
)
list(TRANSFORM lathe_lua_sources PREPEND "${lua_SOURCE_DIR}/")

add_library(lua STATIC ${lathe_lua_sources})
add_library(lua::lua ALIAS lua)

# C on purpose: compiling Lua as C++ makes it throw, which needs -fexceptions in the scripting module and breaks its
# shared PCH. No LUA_USE_POSIX/LINUX: those pull in _setjmp/dlopen/mkstemp, which -std=c17 doesn't declare.
set_target_properties(
    lua
    PROPERTIES
        C_STANDARD 17
        C_STANDARD_REQUIRED YES
        C_EXTENSIONS NO
)
target_include_directories(lua PUBLIC "${lua_SOURCE_DIR}")

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(lua PRIVATE -w)
endif()

if(UNIX)
  target_link_libraries(lua PRIVATE m)
endif()

CPMAddPackage(
    NAME sol2
    GITHUB_REPOSITORY ThePhD/sol2
    GIT_TAG v3.5.0
    DOWNLOAD_ONLY YES
)

# sol2's own CMakeLists needs CMake 3.26 and drags in FetchContent/LuaBuild; the headers are all we need.
add_library(sol2 INTERFACE)
add_library(sol2::sol2 ALIAS sol2)
target_include_directories(sol2 INTERFACE "${sol2_SOURCE_DIR}/include")
target_link_libraries(sol2 INTERFACE lua::lua)

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
# KTX-Software (libktx): KTX2 read/write and the Basis Universal UASTC
# encoder/transcoder. Only the library is needed; we do our own uploads.
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
# etcdec.cxx is under an Ericsson license that is not open source (OpenGL/GLES/WebGL products only). Nothing here needs
# software ETC decoding: Basis textures are transcoded to BC7/BC5.
set(KTX_FEATURE_ETC_UNPACK OFF CACHE BOOL "" FORCE)

# Skip the LFS test images and the CTS submodule; nothing we build reads them.
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

  # imgui_internal.h trips -Wconversion, which -Werror turns into an error.
  mark_target_includes_system(imgui)

  # Paths as laid out at the pinned commit; the header lives next to the example backends.
  set(imgui_notify_include_dir "${ImGuiNotify_SOURCE_DIR}/unixExample/backends")
  set(imgui_notify_icons_dir "${ImGuiNotify_SOURCE_DIR}/unixExample/fonts")

  add_library(imgui_notify INTERFACE)
  target_include_directories(imgui_notify INTERFACE "${imgui_notify_include_dir}" "${imgui_notify_icons_dir}")
  target_link_libraries(imgui_notify INTERFACE imgui)
  mark_target_includes_system(imgui_notify)
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

  if(APPLE)
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
  if(APPLE)
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
mark_target_includes_system(lua)
mark_target_includes_system(sol2::sol2)


include("${CMAKE_SOURCE_DIR}/cmake/shader_push_constant_reflection.cmake")


# ------------------------------------------------------------------------------
# Project sources
#
# Each engine_<module> library lists its own sources in
# src/<module>/CMakeLists.txt. This section covers what's shared and the
# sources built directly into the executable.
# ------------------------------------------------------------------------------

# The global operator new/delete override feeds MemoryTracker for the in-app
# Memory panel. Not built in Release, which links jemalloc on Linux instead.
#
# It's an executable source because the linker may drop an unreferenced
# override from a static archive.
set(lathe_track_memory ON)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(lathe_track_memory OFF)
endif()

# Entry point and one-time Vulkan bring-up; nothing else links them.
set(
    lathe_main_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cxx"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan_bootstrap.cxx"
)

foreach(lathe_main_source ${lathe_main_sources})
  if(NOT EXISTS "${lathe_main_source}")
    message(
        FATAL_ERROR
        "Expected application entry point source at "
        "${lathe_main_source}"
    )
  endif()
endforeach()

if(lathe_track_memory)
  list(
      APPEND lathe_main_sources
      "${CMAKE_CURRENT_SOURCE_DIR}/src/global_new_delete.cxx"
  )
endif()

# Game sources are linked only into the executable, keeping the engine
# libraries game-agnostic.
file(
    GLOB_RECURSE lathe_game_sources
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/game/src/*.cxx"
)

list(
    APPEND lathe_main_sources
    ${lathe_game_sources}
)

# Needs stb_SOURCE_DIR, bullet3_SOURCE_DIR, shader_reflect_generated_dir and
# slang_root from above.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/engine_deps.cmake")

# Defines configure_renderdoc() for src/gpu/CMakeLists.txt.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/renderdoc_config.cmake")

# ------------------------------------------------------------------------------
# Module libraries, in dependency order:
# core < gpu < assets < net < physics < scene < terrain < rendering < serialisation < scripting < app
# ------------------------------------------------------------------------------

add_subdirectory(src/core)
add_subdirectory(src/gpu)
add_subdirectory(src/assets)
add_subdirectory(src/net)
add_subdirectory(src/physics)
add_subdirectory(src/scene)
add_subdirectory(src/terrain)
add_subdirectory(src/rendering)
add_subdirectory(src/serialisation)
add_subdirectory(src/scripting)
add_subdirectory(src/app)

# ------------------------------------------------------------------------------
# Core library
#
# lathe-core is an INTERFACE umbrella over the module libraries so the
# executable and tests don't need the module list.
#
# $<LINK_GROUP:RESCAN,...> acts like --start-group/--end-group: core's
# describe() dispatcher (error_describe.cxx) calls overloads defined in higher
# layers, which a single left-to-right archive pass can't resolve.
# ------------------------------------------------------------------------------

add_library(
    lathe-core
    INTERFACE
)

target_link_libraries(
    lathe-core
    INTERFACE
        "$<LINK_GROUP:RESCAN,engine_app,engine_scripting,engine_serialisation,engine_rendering,engine_terrain,engine_scene,engine_physics,engine_net,engine_assets,engine_gpu,engine_core>"
)

add_library(
    lathe::core
    ALIAS
    lathe-core
)

# ------------------------------------------------------------------------------
# Application
# ------------------------------------------------------------------------------

add_executable(
    lathe
    ${lathe_main_sources}
)

target_link_libraries(
    lathe
    PRIVATE
        lathe::core
        engine_options
)

# Bakes an equirect HDR panorama into a KTX2 cubemap environment (see tools/env_bake/main.cxx).
add_executable(
    lathe-env-bake
    tools/env_bake/main.cxx
)

target_link_libraries(
    lathe-env-bake
    PRIVATE
        lathe::core
        engine_options
)

target_include_directories(
    lathe-env-bake
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_include_directories(
    lathe
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/game/include"
)

target_precompile_headers(
    lathe
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include/core/pch.hxx"
)

# ------------------------------------------------------------------------------
# Platform configuration (link and runtime deployment for the executable)
# ------------------------------------------------------------------------------

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  include(
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/platform_linux.cmake"
  )

  configure_linux_target(
      lathe
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
# linux-native Release links jemalloc, which replaces malloc/free (and so the
# default operator new/delete) through symbol interposition. Needs libjemalloc-dev (see Dockerfile).
# ------------------------------------------------------------------------------

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_BUILD_TYPE STREQUAL "Release")
  find_library(LATHE_JEMALLOC_LIBRARY jemalloc)

  if(LATHE_JEMALLOC_LIBRARY)
    target_link_libraries(
        lathe
        PRIVATE
            "${LATHE_JEMALLOC_LIBRARY}"
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
        TARGET lathe
        POST_BUILD
        COMMAND
            "${CMAKE_COMMAND}" -E copy_directory_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/${asset_directory}"
            "$<TARGET_FILE_DIR:lathe>/${asset_directory}"
        VERBATIM
    )
  endif()
endforeach()

# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------

if(BUILD_TESTING)
  include(
      "${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake"
  )

  add_subdirectory(test)
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
    "C++ exceptions enabled: ${LATHE_ENABLE_EXCEPTIONS}"
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

