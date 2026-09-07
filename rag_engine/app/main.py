"""FastAPI entrypoint: /v1/ingest, /v1/ask, /v1/evaluate, /health."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.config import Settings, get_settings
from app.evaluation.metrics import evaluate_from_request
from app.generation.generator import GroundedGenerator
from app.generation.verifier import CitationVerifier
from app.ingestion.indexer import Indexer
from app.pipeline import RAGService
from app.retrieval.dense import DenseRetriever, get_vector_store
from app.retrieval.embeddings import get_embedder
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.sparse import SparseIndexBase, get_sparse_index
from app.schemas.eval import EvalReport, EvalRunRequest
from app.schemas.ingestion import IngestResponse, IndexStatus
from app.schemas.query import QueryRequest, QueryResponse
from app.utils.errors import RAGError
from app.utils.security import RateLimiter, client_key, require_api_key

logger = logging.getLogger("rag_engine")


def configure_logging(settings: Settings) -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    if settings.log_json:
        formatter = logging.Formatter(
            '{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}'
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


@dataclass
class AppContext:
    """Dependency container shared by all routes (swap-able for tests)."""

    settings: Settings
    embedder: object
    store: object
    sparse: SparseIndexBase
    dense: DenseRetriever
    indexer: Indexer
    reranker: CrossEncoderReranker
    generator: GroundedGenerator
    verifier: CitationVerifier
    rag: RAGService

    @property
    def warm(self) -> dict[str, bool]:
        return {
            "embeddings_loaded": bool(getattr(self.embedder, "loaded", True)),
            "reranker_loaded": self.reranker.loaded,
            "llm_available": self.generator.llm_available,
            "verifier_engine_ready": self.verifier.engine is not None,
        }


def build_context(settings: Settings) -> AppContext:
    embedder = get_embedder(settings)
    store = get_vector_store(settings)
    sparse = get_sparse_index(settings)
    dense = DenseRetriever(embedder=embedder, store=store)
    indexer = Indexer(
        settings=settings,
        embedder=embedder,
        vector_store=store,
        sparse_index=sparse,
    )
    reranker = CrossEncoderReranker(settings)
    generator = GroundedGenerator(settings)
    verifier = CitationVerifier(settings)
    rag = RAGService(
        settings=settings,
        dense=dense,
        sparse=sparse,
        reranker=reranker,
        generator=generator,
        verifier=verifier,
    )
    return AppContext(
        settings=settings,
        embedder=embedder,
        store=store,
        sparse=sparse,
        dense=dense,
        indexer=indexer,
        reranker=reranker,
        generator=generator,
        verifier=verifier,
        rag=rag,
    )


async def _warm_models(ctx: AppContext) -> None:
    if not ctx.settings.warm_models:
        return
    logger.info("Warming models (embeddings=%s, reranker=%s)...",
                ctx.settings.embedding_provider, ctx.settings.cross_encoder_model)

    def _warm_sync() -> None:
        try:
            ctx.embedder.embed(["warmup"])
        except Exception as exc:
            logger.warning("Embedding warmup failed: %s", exc)
        try:
            ctx.reranker.ensure_model()
        except Exception as exc:
            logger.warning("Reranker warmup failed: %s", exc)

    await asyncio.to_thread(_warm_sync)
    with contextlib.suppress(Exception):
        ctx.verifier._resolve_judge()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ctx: AppContext | None = getattr(app.state, "ctx", None)
    if ctx is None:
        ctx = build_context(get_settings())
        app.state.ctx = ctx
    if ctx.settings.environment == "prod" and not ctx.settings.api_key:
        logger.warning(
            "Running in 'prod' with no API_KEY set: /v1/ingest, /v1/ask and /v1/evaluate "
            "are unauthenticated. Set API_KEY to require the X-API-Key header."
        )
    await _warm_models(ctx)
    stats = ctx.indexer.stats()
    logger.info(
        "%s v%s ready | env=%s | index=%s chunks=%s bm25=%s",
        ctx.settings.app_name,
        __version__,
        ctx.settings.environment,
        stats["vector_store"],
        stats["chunk_count"],
        stats["bm25_documents"],
    )
    try:
        yield
    finally:
        with contextlib.suppress(Exception):
            await ctx.generator.aclose()
        logger.info("Shutdown complete")


def create_app(ctx: AppContext | None = None) -> FastAPI:
    app = FastAPI(
        title="Production RAG Engine",
        description=(
            "Hybrid retrieval (dense + BM25), Reciprocal Rank Fusion, cross-encoder "
            "reranking, grounded generation with citation verification, and a statistical "
            "evaluation harness."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    if ctx is not None:
        app.state.ctx = ctx

    resolved_settings = ctx.settings if ctx is not None else get_settings()
    limiter = RateLimiter(max_requests=resolved_settings.rate_limit_per_minute, window_seconds=60.0)

    async def enforce_rate_limit(request: Request) -> None:
        limiter.check(client_key(request))

    protected = [Depends(require_api_key), Depends(enforce_rate_limit)]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_telemetry(request: Request, call_next):
        request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["x-request-id"] = request_id
        response.headers["x-process-time-ms"] = f"{elapsed_ms:.1f}"
        logger.info(
            "%s %s -> %d (%.1f ms) [%s]",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request_id,
        )
        return response

    @app.exception_handler(RAGError)
    async def rag_error_handler(request: Request, exc: RAGError) -> JSONResponse:
        logger.error("Domain error %s: %s", exc.error_code, exc.message)
        app_ctx: AppContext | None = getattr(request.app.state, "ctx", None)
        expose_details = app_ctx is None or app_ctx.settings.environment != "prod"
        detail = exc.message if expose_details else "The request could not be processed."
        return JSONResponse(
            status_code=exc.http_status,
            content={"error": exc.error_code, "detail": detail},
        )

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {"service": "rag-engine", "docs": "/docs", "health": "/health"}

    @app.post(
        "/v1/ingest",
        response_model=IngestResponse,
        tags=["Ingestion"],
        summary="Parse, chunk, deduplicate and index a document (PDF/MD/TXT/HTML)",
        dependencies=protected,
    )
    async def ingest_document(
        http_request: Request,
        file: UploadFile = File(..., description="Document to ingest (.pdf, .md, .txt, .html)"),
        strategy: str | None = Form(
            default=None,
            description="Chunking strategy override: fixed | structure | semantic",
        ),
    ) -> IngestResponse:
        app_ctx: AppContext = http_request.app.state.ctx
        filename = Path(file.filename or "unnamed").name
        suffix = Path(filename).suffix.lower()
        allowed = {".pdf", ".md", ".txt", ".html", ".htm"}
        if suffix not in allowed:
            raise HTTPException(status_code=415, detail=f"Unsupported file type '{suffix}'")

        max_bytes = app_ctx.settings.max_upload_mb * 1024 * 1024
        declared_length = http_request.headers.get("content-length")
        if declared_length is not None:
            try:
                if int(declared_length) > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds {app_ctx.settings.max_upload_mb} MB limit",
                    )
            except ValueError:
                pass

        # Stream-read with a hard cap so a spoofed/absent Content-Length (or chunked
        # transfer encoding) can't force the whole body into memory before rejection.
        chunks: list[bytes] = []
        total = 0
        while True:
            piece = await file.read(1024 * 1024)
            if not piece:
                break
            total += len(piece)
            if total > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload exceeds {app_ctx.settings.max_upload_mb} MB limit",
                )
            chunks.append(piece)
        content = b"".join(chunks)
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        return await app_ctx.indexer.ingest(filename, content, strategy)

    @app.post(
        "/v1/ask",
        response_model=QueryResponse,
        tags=["Query"],
        summary="Full RAG pipeline: hybrid retrieval -> RRF -> rerank -> generation -> citation verification",
        dependencies=protected,
    )
    async def ask(http_request: Request, request: QueryRequest) -> QueryResponse:
        app_ctx: AppContext = http_request.app.state.ctx
        return await app_ctx.rag.ask(request)

    @app.post(
        "/v1/evaluate",
        response_model=EvalReport,
        tags=["Evaluation"],
        summary="Run the benchmark harness (Recall@K, MRR, faithfulness) over a golden dataset",
        dependencies=protected,
    )
    async def evaluate(http_request: Request, request: EvalRunRequest) -> JSONResponse:
        app_ctx: AppContext = http_request.app.state.ctx
        report, json_path, md_path = await evaluate_from_request(
            rag=app_ctx.rag,
            verifier=app_ctx.verifier,
            settings=app_ctx.settings,
            request=request,
        )
        payload = report.model_dump(mode="json")
        payload["artifacts"] = {"json": str(json_path), "markdown": str(md_path)}
        return JSONResponse(status_code=200, content=payload)

    @app.get("/health", tags=["Ops"], summary="Healthcheck with index status and model readiness")
    async def health(http_request: Request) -> JSONResponse:
        app_ctx: AppContext = http_request.app.state.ctx
        stats = app_ctx.indexer.stats()
        body = {
            "status": "ok",
            "version": __version__,
            "environment": app_ctx.settings.environment,
            "index": {
                **stats,
                "embedding_provider": app_ctx.settings.embedding_provider,
                "embedding_model": getattr(app_ctx.embedder, "name", "unknown"),
                "embedding_dim": getattr(app_ctx.embedder, "dim", None),
                "collection_name": app_ctx.settings.collection_name,
            },
            "models": {
                "embeddings_loaded": bool(getattr(app_ctx.embedder, "loaded", True)),
                "reranker_model": app_ctx.settings.cross_encoder_model,
                "reranker_loaded": app_ctx.reranker.loaded,
                "llm_chat_model": app_ctx.settings.openai_chat_model,
                "llm_available": app_ctx.generator.llm_available,
                "verifier_mode": app_ctx.verifier.mode,
                "verifier_engine": app_ctx.verifier.engine,
            },
        }
        degraded = (
            not bool(getattr(app_ctx.embedder, "loaded", True))
            and app_ctx.settings.warm_models
            and app_ctx.settings.embedding_provider == "huggingface"
        )
        if degraded:
            body["status"] = "degraded"
        return JSONResponse(status_code=200, content=body)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "dev",
    )
