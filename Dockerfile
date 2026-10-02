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

ARG SPIRV_TOOLS_TAG=vulkan-sdk-1.4.357.0

RUN git clone --branch "${SPIRV_TOOLS_TAG}" --depth 1 \
        https://github.com/KhronosGroup/SPIRV-Tools.git /tmp/spirv-tools \
    && git clone --branch "${SPIRV_TOOLS_TAG}" --depth 1 \
        https://github.com/KhronosGroup/SPIRV-Headers.git /tmp/spirv-tools/external/spirv-headers \
    && cmake -S /tmp/spirv-tools -B /tmp/spirv-tools/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DSPIRV_SKIP_TESTS=ON \
        -DSPIRV_SKIP_EXECUTABLES=ON \
        -DSPIRV_WERROR=OFF \
    && cmake --build /tmp/spirv-tools/build -j"$(nproc)" \
    && cmake --install /tmp/spirv-tools/build --prefix /usr/local \
    && rm -rf /tmp/spirv-tools

WORKDIR /workspace
