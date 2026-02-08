# Base image for NVIDIA GPUs (ARM64 compatible version)
FROM nvcr.io/nvidia/tensorflow:24.05-tf2-py3

# Install build dependencies for drjit/mitsuba on ARM
RUN apt-get update && apt-get install -y \
    cmake \
    ninja-build \
    g++ \
    libx11-dev \
    libxcursor-dev \
    libxinerama-dev \
    libxrandr-dev \
    libxi-dev \
    libgl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Sionna and common tools
# This may take time on first build as it compiles drjit/mitsuba from source
RUN pip install sionna==1.2.1 matplotlib pandas scipy tqdm

# Set working directory
WORKDIR /mnt/nas_data/sionna-sim
