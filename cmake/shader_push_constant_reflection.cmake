find_program(
    shader_reflect_host_cargo
    NAMES cargo
    HINTS
    "$ENV{CARGO_HOME}/bin"
    "$ENV{HOME}/.cargo/bin"
    NO_CMAKE_FIND_ROOT_PATH
)

set(shader_reflect_cargo_environment)

if(NOT CMAKE_HOST_WIN32)
    find_program(
        shader_reflect_host_c_compiler
        NAMES cc gcc clang
        REQUIRED
    )

    find_program(
        shader_reflect_host_cxx_compiler
        NAMES c++ g++ clang++
        REQUIRED
    )

    list(
        APPEND
        shader_reflect_cargo_environment
        "CC=${shader_reflect_host_c_compiler}"
        "CXX=${shader_reflect_host_cxx_compiler}"
    )
endif()

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
    shader_reflect_manifest
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/shader_reflect/Cargo.toml"
)

set(
    shader_reflect_rust_source
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/shader_reflect/src/main.rs"
)

set(
    shader_reflect_cargo_target_dir
    "${shader_reflect_generated_dir}/cargo/shader_reflect"
)

if(CMAKE_HOST_WIN32)
    set(shader_reflect_host_executable_suffix ".exe")
else()
    set(shader_reflect_host_executable_suffix "")
endif()

set(
    shader_reflect_tool
    "${shader_reflect_cargo_target_dir}/release/reflect_push_constants${shader_reflect_host_executable_suffix}"
)

list(
    APPEND
    shader_reflect_cargo_environment
    "CARGO_TARGET_DIR=${shader_reflect_cargo_target_dir}"
)

add_custom_command(
    OUTPUT
        "${shader_reflect_tool}"

    COMMAND
        "${CMAKE_COMMAND}"
        -E
        make_directory
        "${shader_reflect_cargo_target_dir}"

    COMMAND
        "${CMAKE_COMMAND}"
        -E
        env
        ${shader_reflect_cargo_environment}
        "${shader_reflect_host_cargo}"
        build
        --manifest-path
        "${shader_reflect_manifest}"
        --release
        --locked

    DEPENDS
        "${shader_reflect_manifest}"
        "${shader_reflect_rust_source}"

    COMMENT
        "Building host shader reflection tool"

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
    "${CMAKE_SOURCE_DIR}/cmake/shader_push_constants_preamble.hxx"
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
