# Use an NVIDIA CUDA base image with development tools (needed for potential compilations)
# Choose a CUDA version compatible with H100/Ampere FP8 (>=11.8 recommended, using 12.x here)
# Ensure the base image OS supports Python 3.11 easily (Ubuntu 22.04 is good)
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHON_VERSION=3.11 \
    # Set HF cache directory (optional, but good practice)
    HF_HOME=/root/.cache/huggingface \
    # Prevent pip from complaining about running as root
    PIP_ROOT_USER_ACTION=ignore

# Install system dependencies: git, build tools, Python 3.11, AND cuDNN dev libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        git \
        wget \
        nano \
        build-essential \
        # Add cuDNN development packages for CUDA 12.x on Ubuntu 22.04
        # Adjust package names if needed based on exact CUDA/OS version requirements
        libcudnn8=8.9.7.29-1+cuda12.2 \
        libcudnn8-dev=8.9.7.29-1+cuda12.2 \
        && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-distutils \
        python3-pip && \
    apt-get purge -y --auto-remove software-properties-common && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} && \
    # Verify python version
    python3 --version

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Set the working directory
WORKDIR /app

# Clone the evo2 repository including submodules (like vortex)
# Using HTTPS instead of SSH for easier access in automated builds/different environments
RUN git clone --recurse-submodules https://github.com/ArcInstitute/evo2.git .

# Install evo2 and its dependencies from the cloned repository
# This will trigger the compilation of transformer-engine, which now should find cudnn.h
RUN pip install --no-cache-dir .

RUN pip uninstall -y transformer_engine || true
RUN pip install --no-cache-dir transformer_engine[pytorch]==1.13

# Inside Dockerfile
RUN pip install flash-attn --no-build-isolation

# Verify installation by trying to import (optional)
# RUN python3 -c "import evo2; import transformer_engine; print('evo2 and transformer_engine imported successfully')"

# Set default command to bash for interactive use (optional)
# Or leave it empty to require users to specify a command
CMD ["bash"]

# --- Notes ---
# 1. Hardware Requirement: This container assumes it will be run on a host machine
#    with NVIDIA GPUs having compute capability >= 8.9 (e.g., H100) for full FP8 support.
# 2. Runtime Requirement: You MUST run this container with the --gpus flag, e.g., --gpus all.
# 3. Model Downloads: Evo2 models are downloaded on first use by the library itself
#    (e.g., when you call `Evo2('evo2_7b')`). They are NOT included in the image.
#    Mounting a volume for the Hugging Face cache is recommended to avoid re-downloading.
# 4. cuDNN version: Pinned cuDNN version 8.9.7.29 for CUDA 12.2 compatibility, which is known to work with recent PyTorch/TE versions.
#    You might adjust this based on specific requirements or newer compatible versions found in NVIDIA repos.
