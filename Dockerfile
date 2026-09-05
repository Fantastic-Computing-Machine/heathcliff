FROM python:3.11-slim AS build

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH="/app/.venv/bin:${PATH}"
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY . .
RUN uv sync --frozen --no-dev

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH="/app/.venv/bin:${PATH}"
WORKDIR /app
RUN useradd --create-home --uid 10001 heathcliff
COPY --from=build /app /app
RUN chown -R heathcliff:heathcliff /app
USER heathcliff
EXPOSE 8700
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8700/v2/runtime/health')" || exit 1
CMD ["python", "-m", "ui.runtime_server"]
