"""Evaluation harness: dataset loading, Recall@K / MRR / faithfulness metrics, benchmark runner."""

from app.evaluation.dataset import is_relevant_hit, load_cases
from app.evaluation.metrics import (
    EvalRunner,
    FaithfulnessTracker,
    build_markdown_table,
    recall_at_k,
    reciprocal_rank,
    render_markdown_report,
    save_report,
)
from app.schemas.eval import GoldenCase

__all__ = [
    "load_cases",
    "is_relevant_hit",
    "GoldenCase",
    "recall_at_k",
    "reciprocal_rank",
    "FaithfulnessTracker",
    "EvalRunner",
    "build_markdown_table",
    "save_report",
    "render_markdown_report",
]
