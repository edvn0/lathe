FROM debian:trixie-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    mingw-w64 \
    g++-mingw-w64-x86-64 \
    gcc-mingw-w64-x86-64 \
    cmake \
    ninja-build \
    git \
    pkg-config \
    ca-certificates \
    curl \
    unzip \
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
    && update-alternatives --set x86_64-w64-mingw32-gcc /usr/bin/x86_64-w64-mingw32-gcc-posix \
    && update-alternatives --set x86_64-w64-mingw32-g++ /usr/bin/x86_64-w64-mingw32-g++-posix \
    && rm -rf /var/lib/apt/lists/*

# Installed outside /root: compile.sh runs this image's builds as the
# invoking host UID (not root) whenever the Docker daemon is rootful, so it
# can't write into /root -- and Debian's /root is mode 0700, meaning that
# UID can't even traverse into it to read a root-owned CARGO_HOME/RUSTUP_HOME
# living there. /opt/cargo and /opt/rustup are then chmod'd world-writable
# below so that same UID can populate cargo's registry cache when building
# tools/shader_reflect (see cmake/shader_push_constant_reflection.cmake).
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
