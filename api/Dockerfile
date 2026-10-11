# Ripple API — the bring-your-own-data path.
# Build context is the REPO ROOT, because the API imports ripple_model directly:
#   docker build -f api/Dockerfile -t ripple-api .
FROM python:3.12-slim

# uv for a fast, lockfile-exact install
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/usr/local \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEMO_MODE=1

WORKDIR /app

# PyMC needs a C compiler at runtime: pytensor compiles the model graph on first use
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

# dependencies first, so a code change does not reinstall PyMC
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project --no-dev

COPY model/ ./model/
COPY api/ ./api/
COPY contract/ ./contract/
RUN uv sync --locked --no-dev

# a non-root user, since this endpoint accepts file uploads
RUN useradd --create-home --uid 10001 ripple && chown -R ripple:ripple /app
USER ripple

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
