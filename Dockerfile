FROM python:3.9-slim

WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-docker.txt ./requirements.txt

# Install Python dependencies (optimized for Docker)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p data/uploads data/processed data/history

# Expose the port Streamlit runs on
EXPOSE 7860

# Health check
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health

# Run the Streamlit app
ENTRYPOINT ["streamlit", "run", "app_enhanced.py", "--server.port=7860", "--server.address=0.0.0.0"]