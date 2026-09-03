# ------------------------------------------------------------------------------
# engine_deps: shared third-party dependency environment for every module
# library (include dirs, link libraries, compile definitions).
#
# These are intentionally PUBLIC/INTERFACE across the board -- see the note
# in CMakeLists.txt above the "Core library" section about the initial
# core-library refactor. Individual libraries can later be tightened to
# PRIVATE where no public project header exposes a given dependency.
#
# Must be included after every CPMAddPackage() call it references, and after
# the shader-reflect setup (needs shader_reflect_generated_dir) and the Slang
# resolution (needs slang_root).
# ------------------------------------------------------------------------------

add_library(engine_deps INTERFACE)

target_include_directories(
    engine_deps
    INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${shader_reflect_generated_dir}/include"
)

# stb is DOWNLOAD_ONLY (no CMake target of its own) and bullet3's own
# CMakeLists uses directory-scoped INCLUDE_DIRECTORIES() rather than
# target_include_directories() (so its include path never reaches downstream
# targets via target_link_libraries()) -- both need their path added
# directly, with SYSTEM requested here since mark_target_includes_system()
# has nothing to act on for either.
target_include_directories(
    engine_deps
    SYSTEM
    INTERFACE
        "${stb_SOURCE_DIR}"
        "${bullet3_SOURCE_DIR}/src"
)

target_link_libraries(
    engine_deps
    INTERFACE
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
        tinyexr
        ktx
        BS_thread_pool
        TracyClient
)

target_compile_definitions(
    engine_deps
    INTERFACE
        VK_NO_PROTOTYPES
        GLFW_INCLUDE_NONE
        GLM_FORCE_RADIANS
        GLM_FORCE_DEPTH_ZERO_TO_ONE
        GLM_ENABLE_EXPERIMENTAL
        SLANG_ROOT_PATH="${slang_root}"
        MINGW_VULKAN_ENABLE_VALIDATION=$<BOOL:${MINGW_VULKAN_ENABLE_VALIDATION}>
)
