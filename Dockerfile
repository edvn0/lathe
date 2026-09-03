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

ENV CARGO_HOME=/root/.cargo
ENV RUSTUP_HOME=/root/.rustup
ENV PATH="${CARGO_HOME}/bin:${PATH}"

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain stable \
    && cargo --version \
    && rustc --version

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
