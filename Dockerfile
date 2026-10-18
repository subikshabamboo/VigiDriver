# VigiDriver Production Container
# Compatible with: Hugging Face Spaces (Docker), Render, Railway, GCP, AWS, Azure, Local Docker
FROM python:3.11-slim

# Install system dependencies for OpenCV Headless and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Upgrade pip and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy built frontend assets, trained neural models, and backend code
COPY dist/ ./dist/
COPY models/ ./models/
COPY backend/ ./backend/

# Hugging Face Spaces requires non-root user with UID 1000
RUN useradd -m -u 1000 user && \
    chown -R user:user /app
USER user

# Runtime environment settings
ENV HOME=/home/user \
    PORT=7860 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# Expose default port (7860 for Hugging Face Spaces, Render dynamically injects $PORT)
EXPOSE 7860

# Launch all-in-one FastAPI application (serves API + React frontend)
CMD ["sh", "-c", "python -m uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
