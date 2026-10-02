# The tools tools/check_shaders.py runs and nothing else: slangc, spirv-opt and spirv-val. CI's `shaders` job runs in
# this image instead of the build image, so a shader check needs no compiler toolchain and no C++ build, and the image
# is a fraction of the size. Published by .github/workflows/toolchain-image.yml like the build image, tagged with this
# file's hash.
#
# Keep the versions equal to the build's: SLANG_VERSION is CMakeLists.txt's and SPIRV_TOOLS_TAG is the build
# Dockerfile's. The `shaders` job fails when they drift, so it checks the shaders with the tools the engine ships.

ARG SLANG_VERSION=2026.12.0.1
ARG SPIRV_TOOLS_TAG=vulkan-sdk-1.4.357.0

# SPIRV-Tools with its executables, which the build image skips: it only needs the libraries. Release and the same
# base as the build image, so the optimizer is the one the engine links.
FROM debian:trixie-slim AS spirv-tools

ARG SPIRV_TOOLS_TAG
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git ca-certificates python3 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --branch "${SPIRV_TOOLS_TAG}" --depth 1 \
        https://github.com/KhronosGroup/SPIRV-Tools.git /spirv-tools \
    && git clone --branch "${SPIRV_TOOLS_TAG}" --depth 1 \
        https://github.com/KhronosGroup/SPIRV-Headers.git /spirv-tools/external/spirv-headers \
    && cmake -S /spirv-tools -B /spirv-tools/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DSPIRV_SKIP_TESTS=ON \
        -DSPIRV_WERROR=OFF \
    && cmake --build /spirv-tools/build --target spirv-opt spirv-val -j"$(nproc)" \
    && strip /spirv-tools/build/tools/spirv-opt /spirv-tools/build/tools/spirv-val

# The Slang release, cut down to what compiling to SPIR-V needs. `-O3` runs Slang's bundled spirv-opt, which it loads
# from libslang-glslang; libslang-llvm (CPU targets, ~150 MB) and the rest of the release are left out.
FROM debian:trixie-slim AS slang

ARG SLANG_VERSION
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /release /slang \
    && curl --proto '=https' --tlsv1.2 -fsSL \
        "https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/slang-${SLANG_VERSION}-linux-x86_64.tar.gz" \
        | tar -xz -C /release \
    && slangc_path="$(find /release -name slangc -type f | head -n 1)" \
    && root="$(dirname "$(dirname "${slangc_path}")")" \
    && mkdir -p /slang/bin /slang/lib \
    && cp -a "${root}/bin/slangc" /slang/bin/ \
    && cp -a "${root}"/lib/libslang-compiler.so* "${root}"/lib/libslang-glslang-*.so /slang/lib/

FROM debian:trixie-slim

ENV DEBIAN_FRONTEND=noninteractive

# python3: tools/check_shaders.py. slangc is found through its rpath ($ORIGIN/../lib).
RUN apt-get update && apt-get install -y --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=slang /slang /opt/slang
COPY --from=spirv-tools /spirv-tools/build/tools/spirv-opt /spirv-tools/build/tools/spirv-val /usr/local/bin/

ENV PATH="/opt/slang/bin:${PATH}"

# Fails the image build, not the first CI run, if a library slangc needs was left out: compile a trivial shader with
# -O3, then optimize and validate it.
RUN slangc -v && spirv-opt --version && spirv-val --version \
    && printf '[shader("compute")]\n[numthreads(1, 1, 1)]\nvoid main_cs() {}\n' > /tmp/smoke.slang \
    && slangc /tmp/smoke.slang -entry main_cs -stage compute -target spirv -profile spirv_1_6 -O3 -o /tmp/smoke.spv \
    && spirv-opt --target-env=vulkan1.3 -O /tmp/smoke.spv -o /tmp/smoke.opt.spv \
    && spirv-val --target-env vulkan1.3 /tmp/smoke.opt.spv \
    && rm /tmp/smoke.*

WORKDIR /workspace
