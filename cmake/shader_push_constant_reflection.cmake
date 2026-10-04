find_program(
    shader_reflect_host_cargo
    NAMES cargo
    HINTS
    "$ENV{CARGO_HOME}/bin"
    "$ENV{HOME}/.cargo/bin"
    NO_CMAKE_FIND_ROOT_PATH
    REQUIRED
)

set(shader_reflect_cargo_environment)

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

set(
    shader_reflect_slangc
    "${slang_bin_dir}/slangc"
)

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

# CI points this at a directory that outlives the build tree, so the crates (SPIRV-Cross among them) aren't rebuilt
# for every fresh build. Builds of differing tools/shader_reflect sources must not share one.
set(
    LATHE_CARGO_TARGET_DIR
    ""
    CACHE PATH
    "Cargo target directory for tools/shader_reflect (empty: inside the build tree)"
)

if(LATHE_CARGO_TARGET_DIR)
    set(
        shader_reflect_cargo_target_dir
        "${LATHE_CARGO_TARGET_DIR}"
    )
else()
    set(
        shader_reflect_cargo_target_dir
        "${shader_reflect_generated_dir}/cargo/shader_reflect"
    )
endif()

set(
    shader_reflect_tool
    "${shader_reflect_cargo_target_dir}/release/reflect_push_constants"
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

# add_shader_push_constant(
#     <slang file>
#     <entry point>
#     <stage>
#     <generated struct name>
# )
#
# Each shader compiles separately so Ninja can parallelise; the SPIR-V files
# are then reflected together in one process.
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
    main_task
    amplification
    ForwardPushConstants
)

add_shader_push_constant(
    shadow_depth.slang
    main_task
    amplification
    ShadowPushConstants
)

add_shader_push_constant(
    composite.slang
    main_fs
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
    main_cs
    compute
    CullPushConstants
)

add_shader_push_constant(
    instance_lod.slang
    main_cs
    compute
    InstanceLodPushConstants
)

add_shader_push_constant(
    bloom_downsample.slang
    main_cs
    compute
    DownsamplePushConstants
)

add_shader_push_constant(
    bloom_upsample.slang
    main_cs
    compute
    UpsamplePushConstants
)

add_shader_push_constant(
    gtao.slang
    main_cs
    compute
    GtaoPushConstants
)

add_shader_push_constant(
    gtao_denoise.slang
    main_cs
    compute
    GtaoDenoisePushConstants
)

add_shader_push_constant(
    light_cull.slang
    main_cs
    compute
    LightCullPushConstants
)

add_shader_push_constant(
    light_cluster.slang
    main_cs
    compute
    LightClusterPushConstants
)

add_shader_push_constant(
    hiz_build.slang
    main_cs
    compute
    HizBuildPushConstants
)

add_shader_push_constant(
    env_brdf_lut.slang
    main_cs
    compute
    EnvBrdfLutPushConstants
)

add_shader_push_constant(
    env_equirect_to_cube.slang
    main_cs
    compute
    EnvEquirectToCubePushConstants
)

add_shader_push_constant(
    env_sky_to_cube.slang
    main_cs
    compute
    EnvSkyToCubePushConstants
)

add_shader_push_constant(
    env_downsample.slang
    main_cs
    compute
    EnvDownsamplePushConstants
)

add_shader_push_constant(
    env_sh_project.slang
    main_cs
    compute
    EnvShProjectPushConstants
)

add_shader_push_constant(
    env_prefilter.slang
    main_cs
    compute
    EnvPrefilterPushConstants
)

add_shader_push_constant(
    skybox.slang
    main_fs
    fragment
    SkyboxPushConstants
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
