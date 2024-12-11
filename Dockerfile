# Use a multi-stage build for smaller final image
FROM python:3.10-slim AS builder

# Set the working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies in a virtual environment
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt \
    && /opt/venv/bin/pip install --no-cache-dir uvicorn

# Final stage
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /opt/venv /opt/venv
COPY . .

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PORT=8000 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose the port
EXPOSE 8000

# Run as non-root user for security
RUN addgroup --system --gid 1001 appuser \
    && adduser --system --uid 1001 --ingroup appuser appuser
USER appuser

# Start the application
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]