# Global build arguments
ARG PYTHON_VERSION=3.11
ARG DEBIAN_VERSION=bookworm

# ---------------------------------------------------------------------------------------
# Base Stage: Common setup for both Platform and UI
# ---------------------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-${DEBIAN_VERSION} AS base

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set environment variables for uv and python
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install system dependencies required for building Python packages (like PyAudio)
# These are needed during 'uv sync' if wheels are missing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    portaudio19-dev \
    libasound2-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition files
COPY pyproject.toml uv.lock ./

# Install dependencies (creates .venv)
# We use --frozen to strictly follow uv.lock
RUN uv sync --frozen --no-install-project --no-dev

# ---------------------------------------------------------------------------------------
# UI Stage: Streamlit Application
# ---------------------------------------------------------------------------------------
FROM base AS ui

# Copy application code
COPY . .

# Install the project itself (if needed, or just rely on deps)
RUN uv sync --frozen --no-dev

# Install Playwright browsers for dynamic web fetching
RUN uv run playwright install --with-deps chromium

# Streamlit specific configuration
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501

EXPOSE 8501

CMD ["streamlit", "run", "ui/Home.py"]

# ---------------------------------------------------------------------------------------
# Platform Stage: Voice/Audio Agent (Raspberry Pi Native)
# ---------------------------------------------------------------------------------------
FROM base AS platform

# Install runtime audio dependencies specific to the voice agent
# espeak-ng is needed for pyttsx3 (TTS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Install the project itself
RUN uv sync --frozen --no-dev

# Run the voice agent
CMD ["python", "main.py"]