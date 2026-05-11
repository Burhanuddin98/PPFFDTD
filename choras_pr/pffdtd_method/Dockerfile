FROM python:3.11.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies for mesh generation, HDF5, and OpenGL stubs needed
# by some gmsh/visualisation code paths. Match the patch level used by DG/DE.
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    gmsh \
    libhdf5-dev \
    libglu1-mesa \
    libxcursor1 \
    libxinerama1 \
    && rm -rf /var/lib/apt/lists/*

# Copy method package directory
COPY pffdtd_method /app/pffdtd_method

# Install the method package
RUN pip install --no-cache-dir /app/pffdtd_method

WORKDIR /app/pffdtd_method

# Default command to run the containerized PFFDTD method
CMD ["python", "-m", "pffdtd_interface"]
