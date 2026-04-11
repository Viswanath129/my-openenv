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
FROM ghcr.io/meta-pytorch/openenv-base:latest
WORKDIR /app

# Ensure uv is available
# (openenv-base usually has it, but if not we can install it, but assuming it's available)

# Upgrade pip (fallback/standard) and install dependencies using uv for extreme speed
RUN pip install --no-cache-dir --upgrade pip
RUN pip install uv

# Copy everything
COPY . /app/

# Install the project and dependencies in one go using uv into the system python
RUN uv pip install --system --no-cache .

# Copy built frontend
COPY --from=frontend-builder /frontend/dist /app/frontend/dist

# Ensure icon is present
COPY InboxIQ.png /app/frontend/dist/InboxIQ.png

# Hugging Face default port is 7860
ENV PORT=7860
EXPOSE 7860

# Ensure modules are findable
ENV PYTHONPATH=/app

# Health check per documentation checklist
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Run the server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
