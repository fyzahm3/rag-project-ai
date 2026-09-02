"""Golden dataset loading for the evaluation harness."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from app.schemas.eval import GoldenCase
from app.utils.errors import EvaluationError
from app.utils.text import tokenize

logger = logging.getLogger(__name__)


def load_cases(path: Path) -> list[GoldenCase]:
    """Load a golden dataset from JSONL (one case per line) or a JSON array file."""
    if not path.exists():
        raise EvaluationError(f"Benchmark dataset not found at {path}")
    cases: list[GoldenCase] = []
    try:
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        cases.append(GoldenCase.model_validate_json(line))
                    except Exception as exc:
                        raise EvaluationError(f"Invalid golden case at line {line_number}: {exc}") from exc
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            items = payload if isinstance(payload, list) else payload.get("cases", [])
            cases = [GoldenCase.model_validate(item) for item in items]
    except EvaluationError:
        raise
    except Exception as exc:
        raise EvaluationError(f"Failed to load benchmark dataset {path}: {exc}") from exc

    if not cases:
        raise EvaluationError(f"Benchmark dataset at {path} contains no cases")
    logger.info("Loaded %d golden cases from %s", len(cases), path)
    return cases


def _normalized(text: str) -> str:
    return " ".join(text.lower().split())


def is_relevant_hit(case: GoldenCase, chunk_id: str, chunk_text: str) -> bool:
    """A chunk counts as relevant when its id is listed or it contains a gold snippet."""
    if case.relevant_chunk_ids and chunk_id in case.relevant_chunk_ids:
        return True
    normalized_chunk = _normalized(chunk_text)
    for snippet in case.relevant_snippets:
        normalized_snippet = _normalized(snippet)
        if not normalized_snippet:
            continue
        if normalized_snippet in normalized_chunk or normalized_chunk in normalized_snippet:
            return True
        snippet_tokens = set(tokenize(snippet))
        chunk_tokens = set(tokenize(chunk_text))
        if snippet_tokens and len(snippet_tokens & chunk_tokens) / len(snippet_tokens) >= 0.9:
            return True
    return False


def relevant_ids_match(case: GoldenCase, retrieved_ids: list[str]) -> bool:
    return any(chunk_id in set(case.relevant_chunk_ids) for chunk_id in retrieved_ids)
