# ──────────────────────────────────────────────
# InboxIQ — Dockerfile for Hugging Face Spaces
# OpenEnv-compliant containerized RL environment
# ──────────────────────────────────────────────

# --- Frontend Build Stage ---
FROM node:20-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# --- Backend Stage ---
FROM python:3.10
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything
COPY . /app/

# Install the project
RUN pip install .

# Copy built frontend
COPY --from=frontend-builder /frontend/dist /app/frontend/dist

# Ensure icon is present
COPY InboxIQ.png /app/frontend/dist/InboxIQ.png

# Hugging Face default port is 7860
ENV PORT=7860
EXPOSE 7860

# Ensure modules are findable
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/state')" || exit 1

# Run the server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
