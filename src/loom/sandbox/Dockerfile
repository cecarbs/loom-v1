# loom sandbox image (docs/05 "docker"). Built by `loom sandbox setup --docker`.
# Runs with --network none, a read-only root, and the host user's uid; the workspace and the
# session's sandbox dir are mounted at their host paths.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ripgrep jq curl procps \
    && rm -rf /var/lib/apt/lists/*

ARG PYTHON_PACKAGES="pandas numpy pyarrow"
RUN pip install --no-cache-dir ${PYTHON_PACKAGES}

ENV PYTHONDONTWRITEBYTECODE=1
