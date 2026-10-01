# ------------------------------------------------------------------------------
# engine_deps: third-party include dirs, libraries and definitions shared by
# every module library.
#
# Include after every package it references, the shader-reflect setup and the
# Slang resolution.
# ------------------------------------------------------------------------------

add_library(engine_deps INTERFACE)

target_include_directories(
    engine_deps
    INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${shader_reflect_generated_dir}/include"
)

# stb has no target and bullet3 uses directory-scoped include directories, so
# both paths are added here directly.
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
        lua::lua
        sol2::sol2
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
        LATHE_ENABLE_VALIDATION=$<BOOL:${LATHE_ENABLE_VALIDATION}>
        # One sol2 configuration for every TU (ODR). SOL_NO_EXCEPTIONS is left to sol2's own __EXCEPTIONS detection
        # so LATHE_ENABLE_EXCEPTIONS=ON still works.
        SOL_ALL_SAFETIES_ON=1
        SOL_PRINT_ERRORS=0
        SOL_USING_CXX_LUA=0
)
