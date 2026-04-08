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

# Copy everything for the environment
COPY . /app/

# Install the current project in editable mode if needed, or just ensure paths
RUN pip install -e .

# Copy built frontend
COPY --from=frontend-builder /frontend/dist /app/frontend/dist

# Ensure icon is in the built frontend dist
COPY InboxIQ.png /app/frontend/dist/InboxIQ.png

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/state')" || exit 1

# Run the server using the entry point defined in pyproject.toml or direct uvicorn
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
