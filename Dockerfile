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
FROM python:3.10-slim
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment code
COPY src/ /app/src/
COPY openenv.yaml /app/openenv.yaml
COPY InboxIQ.png /app/InboxIQ.png
COPY inference.py /app/inference.py

# Copy dataset into the container
COPY dataset/ /app/dataset/

# Copy built frontend
COPY --from=frontend-builder /frontend/dist /app/frontend/dist

# Ensure icon is in the built frontend dist (belt-and-suspenders)
COPY InboxIQ.png /app/frontend/dist/InboxIQ.png

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/state')" || exit 1

# Run the server
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
