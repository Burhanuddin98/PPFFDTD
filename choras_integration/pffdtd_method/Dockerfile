# Build context = simulation-backend/ root. Build with:
#   docker build -t pffdtd_method_image -f pffdtd_method/Dockerfile .
# Pinned to the exact patch level used by DG/DE (Python 3.11.13-slim).

FROM python:3.11.13-slim

WORKDIR /app

# System deps for gmsh + HDF5 + scientific stack
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    gmsh \
    libhdf5-dev \
    libglu1-mesa \
    libxcursor1 \
    libxinerama1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY pffdtd_method/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Method interface + shared CHORAS helpers + canonical test geometry
COPY pffdtd_method/PFFDTDInterface.py .
COPY pffdtd_method/iso_metrics.py .
COPY common/MeasurementRoom.geo .
COPY common/MeasurementRoom.msh .
COPY common/HelperFunctions.py .
COPY common/simulation_method_interface.py .

# Cache directory for trained ROMs (persisted between simulations via volume mount)
RUN mkdir -p /app/cache

# Default entry: read JSON_PATH env var, run, POST results back
CMD ["python", "PFFDTDInterface.py"]
