# Production RAG Engine

End-to-end Retrieval-Augmented Generation system: **Hybrid Search → Reciprocal Rank Fusion → Cross-Encoder Reranking → Grounded Generation → Citation Verification**, wrapped in an async FastAPI service with a statistical evaluation harness.

```
                        ┌──────────────────────────────────────────────────────┐
 PDF/MD/TXT/HTML ──►    │ Ingestion                                            │
                        │  parser.py ─► chunker.py ─► dedup(τ=0.95) ─► indexer │
                        └──────────────┬───────────────────────┬───────────────┘
                                       ▼                       ▼
                              ChromaDB (cosine)        BM25 (rank-bm25)
                                       └─────────┬─────────────┘
                                                 ▼
                        ┌──────────────────────────────────────────────────────┐
 Query ────────────────►│ Retrieval: dense@10 ∥ sparse@10                      │
                        │   RRF fusion (k=60, w_dense=.7, w_sparse=.3)         │
                        │   top-20 pool ─► cross-encoder rerank ─► top-K       │
                        └──────────────┬───────────────────────────────────────┘
                                       ▼
                        ┌──────────────────────────────────────────────────────┐
                        │ Generation: strict-grounding prompt, [N] citations    │
                        │ Verification: per-claim entailment judge              │
                        │   (LLM-judge ▸ NLI cross-encoder ▸ heuristic)         │
                        │ Confidence = f(mean relevance, citation pass rate)    │
                        └──────────────────────────────────────────────────────┘
```

## Quickstart

### Local (Python 3.11+)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                  # add OPENAI_API_KEY if you have one

python scripts/seed_data.py           # seeds sample docs + writes data/reports/eval_*.{json,md}
uvicorn app.main:app --reload         # http://localhost:8000/docs
```

> Default embedding provider is local HuggingFace (`all-MiniLM-L6-v2`), so the stack runs **without any API key** — generation falls back to deterministic extractive answering and verification to a lexical entailment heuristic. Set `EMBEDDING_PROVIDER=openai` + `OPENAI_API_KEY` for full LLM quality end-to-end. If you change providers, wipe `data/chroma` (embedding dimensions differ).

### Docker

```bash
docker compose up --build
curl localhost:8000/health
```

The compose file mounts `./data` for ChromaDB persistence + reports, runs the seed script on first boot (`--if-empty`), and ships a container-level healthcheck with a long start period for model downloads.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/v1/ingest` | POST | Multipart upload (`.pdf`, `.md`, `.txt`, `.html`) + optional `strategy` form field (`fixed`\|`structure`\|`semantic`). Parses → chunks → dedups → dual-indexes. |
| `/v1/ask` | POST | Full pipeline with ms telemetry: `{query, top_k, enable_reranking}` → answer, citations, confidence, verification report. |
| `/v1/evaluate` | POST | Runs the benchmark harness over a golden JSONL dataset; returns Recall@K, MRR, faithfulness per strategy; persists JSON + Markdown artifacts. |
| `/v1/find?q=` | GET | Fast file search: dense ∥ sparse (no rerank, no LLM), filename/path matches boosted heavily. Returns `{path, snippet, score, mtime}[]`. Works under both profiles; the filename boost is FTS5-only (local profile) and degrades to a plain content search on BM25 (server profile). |
| `/v1/open` | POST | **Local profile only.** `{path}` → opens that file with the OS default handler. Loopback-only (403 otherwise) and restricted to files the crawler has actually indexed (404 otherwise) — not a general local file/app launcher. |
| `/health` | GET | Index status (chunk counts), embedding model/dim, reranker & LLM readiness. |

```bash
curl -F "file=@data/raw_docs/kubernetes_platform.md" localhost:8000/v1/ingest
curl -X POST localhost:8000/v1/ask -H 'content-type: application/json' \
     -d '{"query": "How fast are volumes provisioned?", "top_k": 5}'
curl -X POST localhost:8000/v1/evaluate -H 'content-type: application/json' \
     -d '{"include_generation": true}'
```

## Local mode (tray daemon)

The same codebase runs a second way: a lightweight daemon that watches folders on your
own machine, keeps them indexed, and gives you instant search from a system tray icon
— no server, no Docker, no OpenAI key required. It's a config **profile**
(`PROFILE=local`), not a fork: it reuses `RAGService`/`Indexer` exactly as the server
does, with local-friendly defaults (on-device embeddings, a SQLite FTS5 sparse index
instead of the in-memory BM25 one, a local Ollama model for generation) swapped in
underneath. `PROFILE=server` (the default) is completely unaffected — everything
local-mode-specific lives under `app/local/` plus a handful of profile-gated defaults
in `app/config.py`.

### macOS — copy-paste setup

```bash
git clone https://github.com/fyzahm3/rag-project-ai.git
cd rag-project-ai/rag_engine
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-local.txt

# optional: local generation via Ollama instead of the extractive fallback
brew install ollama && ollama pull qwen2.5:7b

export PROFILE=local
python scripts/run_local_daemon.py
```

### Windows (PowerShell) — copy-paste setup

```powershell
git clone https://github.com/fyzahm3/rag-project-ai.git
cd rag-project-ai\rag_engine
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-local.txt

# optional: local generation via Ollama instead of the extractive fallback
winget install Ollama.Ollama
ollama pull qwen2.5:7b

$env:PROFILE = "local"
python scripts\run_local_daemon.py
```

On first run this generates a commented, editable config file — open it, uncomment
`watched_paths`, point it at your folders, and restart:

- macOS: `~/Library/Application Support/RagSearch/config.yaml`
- Windows: `%LOCALAPPDATA%\RagSearch\config.yaml`

(Its default already watches `~/Documents` and `~/Desktop`, so the daemon has
something to index even before you edit it.) Click the tray icon → **Search…** for a
small always-on-top window: type a query, press Enter, double-click a result to open
the source file in its default app.

The same search is also available as a plain web page — the daemon runs a FastAPI
server under the hood purely to serve it, at `http://127.0.0.1:8000` by default. Open
it in a browser for a one-box, as-you-type search (`app/static/index.html`, vanilla
JS, no build step) hitting `GET /v1/find`; each result has an **Open** button that
calls `POST /v1/open` to launch the file. `GET /` only serves this page under
`PROFILE=local` — under the server profile it's unchanged (a small JSON status blob).

### What's different under `PROFILE=local`

| | server (default) | local |
|---|---|---|
| Embeddings | configurable, defaults to a local HF model | local HF model `BAAI/bge-small-en-v1.5`, device auto-detected (`mps` on Apple Silicon, `cuda` if available, else `cpu`) |
| Sparse index | in-memory BM25 (rank-bm25), JSONL-persisted | SQLite FTS5 — built for frequent incremental insert/update/delete as files change, with a `files` table tracking content hashes so an unchanged file is never needlessly re-embedded |
| Generation | OpenAI (needs `OPENAI_API_KEY`) or extractive fallback | local Ollama model (`qwen2.5:7b` by default) first; automatically prefers OpenAI instead if `OPENAI_API_KEY` is set; falls back to extractive if neither answers |
| Data location | `./data` (repo-relative) | per-OS app data dir (`~/Library/Application Support/RagSearch` / `%LOCALAPPDATA%\RagSearch`) |
| Config source | `.env` / env vars | `config.yaml` in that same app data dir (auto-generated), env vars still work as overrides |
| Auth / rate limiting | `API_KEY` + per-client limiter guard the HTTP routes | not applicable — no HTTP listener; `api_host` still defaults to `127.0.0.1` for anything that does bind a port |
| Excluded from scanning | — | `node_modules`, `.git`, `venv`/`.venv`, `Library`, `AppData`, `*.app`/`*.exe`/`*.dll`, common archives, and anything over `max_index_file_size_mb` (default 25 MB) |
| Chunk ids | content hash (`sha1(doc, heading, seq, text)`) | `sha256(path)[:16]:seq` — stable across re-indexing an edited file, and lets a whole file's chunks be identified from its path alone |

The crawl (`app/ingestion/watcher.py`) walks every `watched_paths` root on startup,
skipping any file whose (mtime, size) already match what's indexed — so a re-crawl of
an untouched folder costs a `stat()` per file, not a re-read/re-embed — and logs
progress every 100 files so a first run over a large folder isn't silent. Indexing
runs with bounded concurrency (`max_concurrent_indexing`, default 2) so the crawl
doesn't peg the CPU. Live edits are picked up by `watchdog` with a true per-path
debounce: a burst of saves to the same file cancels and restarts a single pending
reindex rather than firing one per event.

Any of these can be overridden explicitly (env var, `.env`, or a value in
`config.yaml`) without losing the rest of the local defaults — e.g. setting
`EMBEDDING_PROVIDER=openai` under `PROFILE=local` keeps everything else local (FTS5,
Ollama, app-data paths) and only swaps the embedding backend.

Notes:
- `requirements-local.txt` (watchdog, pystray, Pillow, PyYAML) is separate from
  `requirements.txt` so the server/Docker image is completely untouched by any of
  this — verify with `git diff` against a clean checkout that `Dockerfile`,
  `docker-compose.yml`, and `requirements.txt` are unchanged by local mode.
- The evaluation harness (`/v1/evaluate`, `scripts/seed_data.py`) always runs under
  the server profile's BM25 sparse index, so Recall@K/MRR benchmark numbers stay
  comparable across runs regardless of whether local mode has ever been used.

## Security notes

- **Auth** — `/v1/ingest`, `/v1/ask` and `/v1/evaluate` are open by default (handy for local dev) but accept an `X-API-Key` header check the moment `API_KEY` is set in the environment. Always set it before exposing the service beyond localhost; the server logs a startup warning if `ENVIRONMENT=prod` with no key configured.
- **Rate limiting** — the same three routes are covered by a per-client, per-process sliding-window limiter (`RATE_LIMIT_PER_MINUTE`, default 30/min). It's in-memory and per-process, so put a real edge limiter in front if you run multiple replicas.
- **Uploads** — `/v1/ingest` rejects oversized bodies via `Content-Length` where present and enforces `MAX_UPLOAD_MB` while streaming, so an oversized or Content-Length-less upload can't be buffered fully into memory before being rejected.
- **Evaluation dataset path** — `dataset_path` in `/v1/evaluate` is constrained to resolve inside `DATA_DIR`; absolute paths or `../` traversal outside it are rejected.
- **Prompt injection** — because `/v1/ingest` accepts arbitrary documents that later get fed into the LLM's context window, the system prompt explicitly instructs the model to treat context blocks as untrusted data, not instructions. This reduces but does not eliminate the risk — gate ingestion behind `API_KEY` in any environment where untrusted parties could reach it.
- **Error responses** — outside `ENVIRONMENT=prod`, API errors include the underlying exception message to speed up debugging. In `prod` they're replaced with a generic message; details still go to the server log.

## Design notes

- **Chunking** — three switchable strategies: `FixedChunker` (sliding window + overlap), `StructureAwareChunker` (markdown/HTML/TXT heading sections), `SemanticBoundaryChunker` (splits where cosine distance between consecutive sentence embeddings exceeds a threshold).
- **Near-duplicate pruning** — candidate chunk embeddings are checked against the existing index *and* the current batch; cosine similarity > `DEDUP_SIMILARITY_THRESHOLD` (0.95) is skipped.
- **Hybrid retrieval** — dense (ChromaDB, `hnsw:space=cosine`) and sparse (BM25Okapi over a persisted tokenized corpus) run concurrently; RRF fuses them rank-wise: `Σ wₘ/(k + rankₘ(d))`.
- **Reranking** — the top-20 fused candidates are scored by `cross-encoder/ms-marco-MiniLM-L-6-v2`; top-K survive.
- **Grounded generation** — system prompt enforces context-only answering, inline `[N]` citations, and a structured `INSUFFICIENT_INFORMATION:` fallback.
- **Citation verification** — every cited sentence is judged against its cited chunks via LLM-as-judge (JSON-constrained), an NLI cross-encoder (`nli-deberta-v3-small`), or a lexical heuristic fallback, producing SUPPORTED / CONTRADICTED / UNVERIFIED verdicts. Composite confidence = retrieval similarity ⊕ verification pass rate − contradiction penalty.
- **Evaluation harness** — compares `dense` vs `sparse` vs `hybrid_rrf` vs `hybrid_rerank`; metrics: Recall@K (snippet/chunk-id matched), MRR of first relevant chunk, groundedness rate of generated claims. Artifacts land in `data/reports/`.

## Tests

```bash
pytest tests/ -q          # offline suite: chunkers, RRF, metrics, verifier, dedup, pipeline, API
```

Tests use dependency-free fakes (hashing embedder, NumPy vector store, heuristic verifier) so no network or model downloads are required.
