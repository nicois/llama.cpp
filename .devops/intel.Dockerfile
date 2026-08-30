ARG ONEAPI_VERSION=2026.1.2-devel-ubuntu24.04

# oneDNN is no longer bundled in the Deep Learning Essentials image as of 2026.1, so
# it is installed explicitly below.  Without it find_package(DNNL) merely emits a
# STATUS message and the build silently loses the oneDNN GEMM and flash-attention
# paths, which are the fast paths on Xe2 (Battlemage) and later.  2026.0.2 is the
# newest oneDNN in the oneAPI apt repo and depends on the 2026.1 DPC++ runtime.
ARG ONEDNN_VERSION=2026.0

ARG BUILD_DATE=N/A
ARG APP_VERSION=N/A
ARG APP_REVISION=N/A

## Build Image

ARG NODE_VERSION=24

FROM docker.io/node:$NODE_VERSION AS web

ARG APP_VERSION

WORKDIR /app/tools/ui

COPY tools/ui/package.json tools/ui/package-lock.json ./
RUN npm ci

COPY tools/ui/ ./
RUN LLAMA_BUILD_NUMBER="$APP_VERSION" npm run build

FROM docker.io/intel/deep-learning-essentials:$ONEAPI_VERSION AS build

ARG GGML_SYCL_F16=ON
ARG LEVEL_ZERO_VERSION=1.28.2
ARG LEVEL_ZERO_UBUNTU_VERSION=u24.04
ARG ONEDNN_VERSION
RUN apt-get update && \
    apt-get install -y git libssl-dev wget ca-certificates \
        intel-oneapi-dnnl-devel-${ONEDNN_VERSION} && \
    cd /tmp && \
    wget -q "https://github.com/oneapi-src/level-zero/releases/download/v${LEVEL_ZERO_VERSION}/level-zero_${LEVEL_ZERO_VERSION}%2B${LEVEL_ZERO_UBUNTU_VERSION}_amd64.deb" -O level-zero.deb && \
    wget -q "https://github.com/oneapi-src/level-zero/releases/download/v${LEVEL_ZERO_VERSION}/level-zero-devel_${LEVEL_ZERO_VERSION}%2B${LEVEL_ZERO_UBUNTU_VERSION}_amd64.deb" -O level-zero-devel.deb && \
    apt-get -o Dpkg::Options::="--force-overwrite" install -y ./level-zero.deb ./level-zero-devel.deb && \
    rm -f /tmp/level-zero.deb /tmp/level-zero-devel.deb

# The image sets CMAKE_PREFIX_PATH for the components it ships; oneDNN is not one of
# them any more, so add it back the same way find_package(DNNL) used to find it.
ENV CMAKE_PREFIX_PATH="/opt/intel/oneapi/dnnl/${ONEDNN_VERSION}/lib/cmake:${CMAKE_PREFIX_PATH}"

WORKDIR /app

COPY . .

COPY --from=web /app/tools/ui/dist tools/ui/dist

RUN if [ "${GGML_SYCL_F16}" = "ON" ]; then \
        echo "GGML_SYCL_F16 is set" \
        && export OPT_SYCL_F16="-DGGML_SYCL_F16=ON" \
        && export SYCL_PROGRAM_COMPILE_OPTIONS="-cl-fp32-correctly-rounded-divide-sqrt"; \
    fi && \
    test -f "/opt/intel/oneapi/dnnl/${ONEDNN_VERSION}/lib/cmake/dnnl/dnnl-config.cmake" \
        || { echo "oneDNN not found; the build would silently disable it" >&2; exit 1; } && \
    echo "Building with dynamic libs" && \
    cmake -B build -DGGML_NATIVE=OFF -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_BUILD_TESTS=OFF ${OPT_SYCL_F16} && \
    cmake --build build --config Release -j$(nproc)

RUN mkdir -p /app/lib && \
    find build -name "*.so*" -exec cp -P {} /app/lib \;

RUN mkdir -p /app/full \
    && cp build/bin/* /app/full \
    && cp *.py /app/full \
    && cp -r conversion /app/full \
    && cp -r gguf-py /app/full \
    && cp -r requirements /app/full \
    && cp requirements.txt /app/full \
    && cp .devops/tools.sh /app/full/tools.sh

FROM docker.io/intel/deep-learning-essentials:$ONEAPI_VERSION AS base

ARG BUILD_DATE=N/A
ARG APP_VERSION=N/A
ARG APP_REVISION=N/A
ARG ONEDNN_VERSION
ARG IMAGE_URL=https://github.com/ggml-org/llama.cpp
ARG IMAGE_SOURCE=https://github.com/ggml-org/llama.cpp
LABEL org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.version=$APP_VERSION \
      org.opencontainers.image.revision=$APP_REVISION \
      org.opencontainers.image.title="llama.cpp" \
      org.opencontainers.image.description="LLM inference in C/C++" \
      org.opencontainers.image.url=$IMAGE_URL \
      org.opencontainers.image.source=$IMAGE_SOURCE

#Following versions are for multiple GPUs, since 26.x has known issue:
#   https://github.com/ggml-org/llama.cpp/issues/21747,
#   https://github.com/intel/compute-runtime/issues/921.
#ARG IGC_VERSION=v2.20.5
#ARG IGC_VERSION_FULL=2_2.20.5+19972
#ARG COMPUTE_RUNTIME_VERSION=25.40.35563.10
#ARG COMPUTE_RUNTIME_VERSION_FULL=25.40.35563.10-0
#ARG IGDGMM_VERSION=22.8.2


ARG IGC_VERSION=v2.34.4
ARG IGC_VERSION_FULL=2_2.34.4+21428
ARG COMPUTE_RUNTIME_VERSION=26.18.38308.1
ARG COMPUTE_RUNTIME_VERSION_FULL=26.18.38308.1-0
ARG IGDGMM_VERSION=22.10.0
RUN mkdir /tmp/neo/ && cd /tmp/neo/ \
  && wget https://github.com/intel/intel-graphics-compiler/releases/download/$IGC_VERSION/intel-igc-core-${IGC_VERSION_FULL}_amd64.deb \
  && wget https://github.com/intel/intel-graphics-compiler/releases/download/$IGC_VERSION/intel-igc-opencl-${IGC_VERSION_FULL}_amd64.deb \
  && wget https://github.com/intel/compute-runtime/releases/download/$COMPUTE_RUNTIME_VERSION/intel-ocloc-dbgsym_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.ddeb \
  && wget https://github.com/intel/compute-runtime/releases/download/$COMPUTE_RUNTIME_VERSION/intel-ocloc_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.deb \
  && wget https://github.com/intel/compute-runtime/releases/download/$COMPUTE_RUNTIME_VERSION/intel-opencl-icd-dbgsym_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.ddeb \
  && wget https://github.com/intel/compute-runtime/releases/download/$COMPUTE_RUNTIME_VERSION/intel-opencl-icd_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.deb \
  && wget https://github.com/intel/compute-runtime/releases/download/$COMPUTE_RUNTIME_VERSION/libigdgmm12_${IGDGMM_VERSION}_amd64.deb \
  && wget https://github.com/intel/compute-runtime/releases/download/$COMPUTE_RUNTIME_VERSION/libze-intel-gpu1-dbgsym_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.ddeb \
  && wget https://github.com/intel/compute-runtime/releases/download/$COMPUTE_RUNTIME_VERSION/libze-intel-gpu1_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.deb \
  && dpkg --install *.deb

# libggml-sycl.so links libdnnl.so, which the image no longer ships (see the
# ONEDNN_VERSION comment at the top), and whose lib directory is therefore missing
# from the image's LD_LIBRARY_PATH.
RUN apt-get update \
    && apt-get install -y libgomp1 curl ffmpeg intel-oneapi-dnnl-${ONEDNN_VERSION} \
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete

ENV LD_LIBRARY_PATH="/opt/intel/oneapi/dnnl/${ONEDNN_VERSION}/lib:${LD_LIBRARY_PATH}"

### Full
FROM base AS full

COPY --from=build /app/lib/ /app
COPY --from=build /app/full /app

WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
        git \
        python3 \
        python3-pip \
        python3-venv && \
    python3 -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    apt autoremove -y && \
    apt clean -y && \
    rm -rf /tmp/* /var/tmp/* && \
    find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete && \
    find /var/cache -type f -delete

ENV PATH="/opt/venv/bin:$PATH"

ENTRYPOINT ["/app/tools.sh"]

### Light, CLI only
FROM base AS light

COPY --from=build /app/lib/ /app
COPY --from=build /app/full/llama /app/full/llama-cli /app/full/llama-completion /app

WORKDIR /app

ENTRYPOINT [ "/app/llama-cli" ]

### Server, Server only
FROM base AS server

ENV LLAMA_ARG_HOST=0.0.0.0

COPY --from=build /app/lib/ /app
COPY --from=build /app/full/llama /app/full/llama-server /app

WORKDIR /app

HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]

ENTRYPOINT [ "/app/llama-server" ]
