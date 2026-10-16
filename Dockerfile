FROM debian:trixie-slim

ENV DEBIAN_FRONTEND=noninteractive

# libvulkan1/mesa-vulkan-drivers: lets the app run inside the container, on lavapipe unless a GPU is passed in.
# zstd: CI jobs run inside this image, and actions/cache compresses with it when present.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    clang-tidy \
    cmake \
    ninja-build \
    git \
    pkg-config \
    ca-certificates \
    curl \
    unzip \
    zstd \
    ccache \
    python3 \
    wayland-protocols \
    libwayland-dev \
    libxkbcommon-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libgl1-mesa-dev \
    libjemalloc-dev \
    mold \
    libcurl4-openssl-dev \
    libvulkan1 \
    mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*

# Outside /root, which is mode 0700: compile.sh runs builds as the host UID with a rootful daemon, and cargo
# needs to write its registry cache when building tools/shader_reflect.
ENV CARGO_HOME=/opt/cargo
ENV RUSTUP_HOME=/opt/rustup
ENV PATH="${CARGO_HOME}/bin:${PATH}"

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain stable \
    && cargo --version \
    && rustc --version \
    && chmod -R a+rwX "${CARGO_HOME}" "${RUSTUP_HOME}"

# Installed with its executables (spirv-opt, spirv-val): tools/check_shaders.py runs them in the ci build job.
ARG SPIRV_TOOLS_TAG=vulkan-sdk-1.4.357.0

RUN git clone --branch "${SPIRV_TOOLS_TAG}" --depth 1 \
        https://github.com/KhronosGroup/SPIRV-Tools.git /tmp/spirv-tools \
    && git clone --branch "${SPIRV_TOOLS_TAG}" --depth 1 \
        https://github.com/KhronosGroup/SPIRV-Headers.git /tmp/spirv-tools/external/spirv-headers \
    && cmake -S /tmp/spirv-tools -B /tmp/spirv-tools/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DSPIRV_SKIP_TESTS=ON \
        -DSPIRV_WERROR=OFF \
    && cmake --build /tmp/spirv-tools/build -j"$(nproc)" \
    && cmake --install /tmp/spirv-tools/build --prefix /usr/local \
    && rm -rf /tmp/spirv-tools

# Assimp, built the way CMakeLists.txt builds it from source (same tag, importers, patch and flags), but shared and
# installed to /usr/local, so the project's find_package(assimp) finds it and CI and local builds link it instead of
# compiling it every time. Keep ASSIMP_TAG, the importer list and the TINYUSDZ define in step with the cpmaddpackage(assimp)
# call there. The tag only changes with this file, so bump ASSIMP_PATCH_REV when cmake/patches/assimp_usd_empty_stage.patch
# changes to publish a new image.
ARG ASSIMP_TAG=v5.4.3
ARG ASSIMP_PATCH_REV=1

COPY cmake/patches/assimp_usd_empty_stage.patch /tmp/assimp_usd_empty_stage.patch

RUN git clone --branch "${ASSIMP_TAG}" --depth 1 https://github.com/assimp/assimp.git /tmp/assimp \
    && git -C /tmp/assimp apply /tmp/assimp_usd_empty_stage.patch \
    && cmake -S /tmp/assimp -B /tmp/assimp/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        "-DCMAKE_C_FLAGS=-DTINYUSDZ_NO_STB_IMAGE_RESIZE_IMPLEMENTATION" \
        "-DCMAKE_CXX_FLAGS=-DTINYUSDZ_NO_STB_IMAGE_RESIZE_IMPLEMENTATION" \
        -DASSIMP_INSTALL=ON \
        -DASSIMP_BUILD_TESTS=OFF \
        -DASSIMP_BUILD_ASSIMP_TOOLS=OFF \
        -DASSIMP_BUILD_SAMPLES=OFF \
        -DASSIMP_NO_EXPORT=ON \
        -DASSIMP_WARNINGS_AS_ERRORS=OFF \
        -DASSIMP_INJECT_DEBUG_POSTFIX=OFF \
        -DASSIMP_BUILD_ZLIB=ON \
        -DASSIMP_BUILD_USE_CCACHE=OFF \
        -DASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT=OFF \
        -DASSIMP_BUILD_FBX_IMPORTER=ON \
        -DASSIMP_BUILD_OBJ_IMPORTER=ON \
        -DASSIMP_BUILD_COLLADA_IMPORTER=ON \
        -DASSIMP_BUILD_PLY_IMPORTER=ON \
        -DASSIMP_BUILD_STL_IMPORTER=ON \
        -DASSIMP_BUILD_USD_IMPORTER=ON \
        -DTINYUSDZ_NO_WERROR=ON \
    && cmake --build /tmp/assimp/build -j"$(nproc)" \
    && cmake --install /tmp/assimp/build --prefix /usr/local \
    && ldconfig \
    && rm -rf /tmp/assimp /tmp/assimp_usd_empty_stage.patch

WORKDIR /workspace
