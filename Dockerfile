FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    libglfw3 \
    libglew2.2 \
    libgomp1 \
    libgfortran5 \
    libquadmath0 \
    liblapack3 \
    libblas3 \
    libosmesa6 \
    libsm6 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

# Optional NASA visual (gitignored): downloaded and hash-checked here; a failed download only logs a warning
COPY scripts/fetch_assets.py scripts/fetch_assets.py
RUN python scripts/fetch_assets.py

COPY . .

RUN useradd --create-home --uid 10001 --user-group simgate \
    && mkdir -p /app/simgate/library/uploads \
    && chown -R simgate:simgate /app

USER simgate

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]