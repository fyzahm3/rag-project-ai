FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/srv/rag_engine/data/hf-cache \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4

WORKDIR /srv/rag_engine

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY . .

RUN useradd --create-home appuser \
    && mkdir -p /srv/rag_engine/data /srv/rag_engine/logs \
    && chown -R appuser:appuser /srv/rag_engine
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).getcode()==200 else 1)"

CMD ["sh", "-c", "python scripts/seed_data.py --if-empty && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
