FROM python:3.10-slim

WORKDIR /app

# System deps for gmsh + HDF5
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglu1-mesa libxcursor1 libxinerama1 libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy PFFDTD submodule + our interface
COPY pffdtd/python /app/pffdtd/python
COPY pffdtd_method /app/pffdtd_method
COPY common /app/common

ENV PYTHONPATH="/app/pffdtd/python:/app/common:/app"

CMD ["python", "/app/pffdtd_method/PFDTDInterface.py"]
