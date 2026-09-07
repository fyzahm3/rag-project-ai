"""Cross-encoder reranking (cross-encoder/ms-marco-MiniLM-L-6-v2 by default)."""
from __future__ import annotations

import asyncio
import logging

import numpy as np

from app.config import Settings
from app.retrieval.dense import RetrievedRecord

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, settings: Settings, model_name: str | None = None) -> None:
        self.settings = settings
        self.model_name = model_name or settings.cross_encoder_model
        self._model = None
        self._load_failed = False

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def ensure_model(self) -> bool:
        if self.loaded:
            return True
        if self._load_failed:
            return False
        try:
            from sentence_transformers import CrossEncoder

            device = None
            if self.settings.profile == "local":
                from app.utils.device import resolve_device

                device = resolve_device(self.settings.local_device)
            logger.info("Loading cross-encoder reranker '%s' (device=%s)", self.model_name, device or "auto")
            self._model = CrossEncoder(self.model_name, max_length=512, device=device)
            return True
        except Exception as exc:
            self._load_failed = True
            logger.warning("Cross-encoder '%s' unavailable (%s); reranking disabled", self.model_name, exc)
            return False

    def rerank_sync(
        self,
        query: str,
        records: list[RetrievedRecord],
        top_k: int,
    ) -> list[RetrievedRecord]:
        if not records:
            return []
        if not self.ensure_model():
            return [r.model_copy(deep=True) for r in records[:top_k]]

        pairs = [[query, record.text[:2000]] for record in records]
        raw_scores = np.asarray(self._model.predict(pairs), dtype=np.float64)
        sigmoid_scores = 1.0 / (1.0 + np.exp(-raw_scores))

        reranked: list[RetrievedRecord] = []
        order = np.argsort(-sigmoid_scores)[:top_k]
        for index in order:
            copy = records[int(index)].model_copy(deep=True)
            copy.score = float(sigmoid_scores[int(index)])
            copy.raw_score = float(raw_scores[int(index)])
            reranked.append(copy)
        return sorted(reranked, key=lambda r: r.score, reverse=True)

    async def rerank(
        self,
        query: str,
        records: list[RetrievedRecord],
        top_k: int,
    ) -> list[RetrievedRecord]:
        return await asyncio.to_thread(self.rerank_sync, query, records, top_k)
