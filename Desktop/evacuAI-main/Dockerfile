# Use official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies required by OpenCV and rendering
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the requirements file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn gevent

# Copy the rest of the application codebase
COPY . .

# Expose the port Azure maps traffic to
EXPOSE 8000

# Run Gunicorn with asynchronous threads to support multiple heavy camera video streams concurrently
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "8", "--worker-connections", "1000", "--timeout", "0", "app:app"]
