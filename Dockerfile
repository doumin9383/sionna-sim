FROM nvcr.io/nvidia/tensorflow:24.10-tf2-py3

# Install Sionna and dependencies automatically
RUN pip install --no-cache-dir sionna matplotlib pandas

# Set working directory to the NAS mount path inside container
WORKDIR /mnt/nas_data/sionna-sim
