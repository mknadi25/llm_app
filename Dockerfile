# Base image
FROM rayproject/ray-ml:2.7.0-py310-cpu

# Set the working directory inside the container
WORKDIR /llm_app

# Install system dependencies and Python dependencies in a single step to optimize build
COPY requirements.txt requirements.txt
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc build-essential \
    && python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install -r requirements.txt --no-cache-dir \
    && python3 -m pip install protobuf==3.20.1 --no-cache-dir \
    && apt-get purge -y --auto-remove gcc build-essential \
    && rm -rf /var/lib/apt/lists/*  # Clean up apt cache to reduce image size

# Copy project files (src and artifacts)
COPY src/ /llm_app/src/
COPY efs/mlflow/ /llm_app/efs/mlflow/

# Expose port for the FastAPI app
EXPOSE 8000

# Entrypoint to run the serve.py application
ENTRYPOINT ["python", "-m", "src.serve"]
# You can pass the run_id as an argument like so
CMD ["--run_id", "94ef9aab9c334010af5aaf553cb3f3d2"]
