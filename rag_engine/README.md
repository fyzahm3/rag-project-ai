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
| `/health` | GET | Index status (chunk counts), embedding model/dim, reranker & LLM readiness. |

```bash
curl -F "file=@data/raw_docs/kubernetes_platform.md" localhost:8000/v1/ingest
curl -X POST localhost:8000/v1/ask -H 'content-type: application/json' \
     -d '{"query": "How fast are volumes provisioned?", "top_k": 5}'
curl -X POST localhost:8000/v1/evaluate -H 'content-type: application/json' \
     -d '{"include_generation": true}'
```

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
