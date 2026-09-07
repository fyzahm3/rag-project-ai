"""Fast file search backing GET /v1/find: dense ∥ sparse (FTS5 filename-boosted
search, or a plain-content BM25 fallback under the server profile), RRF-fused, and
deduplicated to one hit per file — no cross-encoder rerank, no LLM call, so this
stays cheap on a warm index."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from app.config import Settings
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.sparse import SparseIndexBase
from app.schemas.find import FindHit
from app.utils.text import truncate

logger = logging.getLogger(__name__)

_SNIPPET_CHARS = 200


async def find_files(
    dense: DenseRetriever,
    sparse: SparseIndexBase,
    settings: Settings,
    query: str,
) -> list[FindHit]:
    dense_results, sparse_results = await asyncio.gather(
        dense.query(query, settings.find_top_k),
        sparse.find(query, settings.find_top_k),
    )
    fused = reciprocal_rank_fusion(
        [dense_results, sparse_results],
        k=settings.rrf_k,
        weights=[settings.find_dense_weight, settings.find_sparse_weight],
    )

    seen_paths: set[str] = set()
    hits: list[FindHit] = []
    for record in fused:
        if record.document_name in seen_paths:
            continue
        seen_paths.add(record.document_name)
        hits.append(
            FindHit(
                path=record.document_name,
                snippet=truncate(record.text, _SNIPPET_CHARS),
                score=record.score,
                mtime=_resolve_mtime(record.document_name, sparse),
            )
        )
        if len(hits) >= settings.find_result_limit:
            break
    return hits


def _resolve_mtime(path: str, sparse: SparseIndexBase) -> float | None:
    """Prefer a live stat() for freshness (the debounced watcher may lag a very
    recent edit); fall back to the sparse index's own record if the file is gone."""
    try:
        return Path(path).stat().st_mtime
    except OSError:
        pass
    get_file_record = getattr(sparse, "get_file_record", None)
    if get_file_record is not None:
        record = get_file_record(path)
        if record is not None:
            return record.get("mtime")
    return None
