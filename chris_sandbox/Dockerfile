# Dockerfile for AISI Inspect Sandbox Environment
# This creates a secure, isolated environment for agent evaluation

FROM python:3.11-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    bash \
    coreutils \
    curl \
    git \
    vim \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages commonly needed for tasks
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    requests \
    beautifulsoup4 \
    lxml \
    openpyxl \
    python-dateutil

# Create non-root user for security
RUN useradd -m -u 1000 agent && \
    chown -R agent:agent /workspace

# Switch to non-root user
USER agent

# Keep container running
CMD ["tail", "-f", "/dev/null"]
