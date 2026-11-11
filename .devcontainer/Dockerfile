FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        clang-format \
        cmake \
        curl \
        gdb \
        git \
        libprotobuf-dev \
        ninja-build \
        protobuf-compiler \
        python3 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.15 /uv /uvx /usr/local/bin/

ENV UV_PYTHON_PREFERENCE=only-system \
    UV_LINK_MODE=copy

USER ubuntu
WORKDIR /workspaces
