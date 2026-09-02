"""Reciprocal Rank Fusion across ranked result lists."""
from __future__ import annotations

from collections.abc import Sequence

from app.retrieval.dense import RetrievedRecord


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RetrievedRecord]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[RetrievedRecord]:
    """Fuse multiple ranked lists:

        RRF(d) = sum_m w_m / (k + rank_m(d))   with ranks starting at 1.

    The fused value is stored on `raw_score`; `score` is min-max rescaled to (0, 1].
    """
    if not ranked_lists:
        return []
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("Number of fusion weights must match number of ranked lists")

    fused: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    first_seen: dict[str, RetrievedRecord] = {}

    for weight, ranking in zip(weights, ranked_lists):
        for position, record in enumerate(ranking):
            contribution = float(weight) / (k + position + 1)
            fused[record.chunk_id] = fused.get(record.chunk_id, 0.0) + contribution
            current_best = best_rank.get(record.chunk_id)
            if current_best is None or position + 1 < current_best:
                best_rank[record.chunk_id] = position + 1
            if record.chunk_id not in first_seen:
                first_seen[record.chunk_id] = record

    ordered_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)
    max_fused = fused[ordered_ids[0]] if ordered_ids else 1.0

    results: list[RetrievedRecord] = []
    for chunk_id in ordered_ids:
        base = first_seen[chunk_id].model_copy(deep=True)
        fused_score = fused[chunk_id]
        base.raw_score = fused_score
        base.score = fused_score / max_fused if max_fused > 0 else 0.0
        results.append(base)
    return results


def normalize_scores(records: list[RetrievedRecord]) -> list[RetrievedRecord]:
    """Rescale non-negative scores so the leader equals 1.0."""
    if not records:
        return records
    top = max((max(r.score, 0.0) for r in records), default=0.0)
    if top <= 0:
        for record in records:
            record.score = 0.0
        return records
    for record in records:
        record.score = float(min(max(record.score, 0.0) / top, 1.0))
    return records
