"""Retrieval metrics (Recall@K, MRR), faithfulness aggregation, and the benchmark runner."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config import Settings
from app.evaluation.dataset import is_relevant_hit, load_cases
from app.generation.verifier import CitationVerifier, compute_confidence
from app.pipeline import RAGService
from app.schemas.eval import (
    ALL_STRATEGIES,
    EvalReport,
    FaithfulnessStats,
    RetrievalStrategyMetrics,
    StrategyName,
    StrategyReport,
)
from app.schemas.query import SourceCitation
from app.utils.errors import EvaluationError

logger = logging.getLogger(__name__)


def recall_at_k(
    hits_per_case: list[list[bool]],
    total_relevant_per_case: list[int],
    k: int,
) -> float:
    """Mean over cases of |relevant ∩ top-K| / |relevant|; cases with no gold are skipped."""
    ratios: list[float] = []
    for case_hits, total in zip(hits_per_case, total_relevant_per_case):
        if total <= 0:
            continue
        top_k_hits = case_hits[:k]
        ratios.append(sum(1 for hit in top_k_hits if hit) / total)
    if not ratios:
        return 0.0
    return sum(ratios) / len(ratios)


def reciprocal_rank(hits_per_case: list[list[bool]]) -> float:
    """Mean Reciprocal Rank of the first relevant chunk across cases."""
    reciprocals: list[float] = []
    for case_hits in hits_per_case:
        rr = 0.0
        for position, hit in enumerate(case_hits, start=1):
            if hit:
                rr = 1.0 / position
                break
        reciprocals.append(rr)
    if not reciprocals:
        return 0.0
    return sum(reciprocals) / len(reciprocals)


class FaithfulnessTracker:
    def __init__(self) -> None:
        self.total_answers = 0
        self.fully_grounded_answers = 0
        self.total_claims = 0
        self.supported_claims = 0
        self.contradicted_claims = 0
        self.unverified_claims = 0

    def observe(self, verification: dict) -> None:
        counts = verification.get("counts", {})
        claims_total = sum(counts.values())
        self.total_answers += 1
        if claims_total > 0 and counts.get("SUPPORTED", 0) == claims_total:
            self.fully_grounded_answers += 1
        self.total_claims += claims_total
        self.supported_claims += counts.get("SUPPORTED", 0)
        self.contradicted_claims += counts.get("CONTRADICTED", 0)
        self.unverified_claims += counts.get("UNVERIFIED", 0)

    def to_stats(self) -> FaithfulnessStats:
        groundedness_rate = (
            self.supported_claims / self.total_claims if self.total_claims else 0.0
        )
        answer_rate = (
            self.fully_grounded_answers / self.total_answers if self.total_answers else 0.0
        )
        return FaithfulnessStats(
            total_answers=self.total_answers,
            fully_grounded_answers=self.fully_grounded_answers,
            total_claims=self.total_claims,
            supported_claims=self.supported_claims,
            contradicted_claims=self.contradicted_claims,
            unverified_claims=self.unverified_claims,
            groundedness_rate=round(groundedness_rate, 4),
            answer_grounding_rate=round(answer_rate, 4),
        )


class EvalRunner:
    """Runs dense/sparse/hybrid/hybrid+rerank configurations over a golden dataset."""

    def __init__(self, rag: RAGService, verifier: CitationVerifier, settings: Settings) -> None:
        self.rag = rag
        self.verifier = verifier
        self.settings = settings

    async def run(
        self,
        dataset_path: Path,
        ks: list[int] | None = None,
        strategies: list[StrategyName] | None = None,
        include_generation: bool | None = None,
        top_k: int | None = None,
        limit: int | None = None,
    ) -> EvalReport:
        started = time.perf_counter()
        resolved_ks = sorted(set(ks or self.settings.eval_default_ks))
        resolved_strategies = [s for s in (strategies or list(ALL_STRATEGIES)) if s in ALL_STRATEGIES]
        if not resolved_strategies:
            raise EvaluationError("No valid strategies requested")
        generate = (
            include_generation
            if include_generation is not None
            else self.settings.eval_include_generation
        )

        cases = load_cases(dataset_path)
        if limit is not None:
            cases = cases[:limit]

        results: dict[str, StrategyReport] = {}
        max_k = max(max(resolved_ks), top_k or self.settings.rerank_top_k)

        for strategy in resolved_strategies:
            report = await self._run_strategy(cases, strategy, resolved_ks, max_k, top_k, generate)
            results[strategy] = report
            logger.info("Strategy '%s' complete: %s", strategy, report.retrieval.model_dump())

        duration = time.perf_counter() - started
        run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
        report = EvalReport(
            run_id=run_id,
            dataset=str(dataset_path),
            generated_at=datetime.now(timezone.utc),
            duration_seconds=round(duration, 3),
            num_cases=len(cases),
            ks=resolved_ks,
            results=results,
            markdown_table=build_markdown_table(report_results=results, ks=resolved_ks),
        )
        return report

    async def _run_strategy(
        self,
        cases: list,
        strategy: StrategyName,
        ks: list[int],
        max_k: int,
        top_k_override: int | None,
        generate: bool,
    ) -> StrategyReport:
        hits_per_case: list[list[bool]] = []
        totals_per_case: list[int] = []
        tracker = FaithfulnessTracker()

        for case in cases:
            try:
                records = await self.rag.retrieve(strategy=strategy, query=case.question, k=max_k)
            except Exception as exc:
                logger.error("Retrieval failed for strategy=%s question=%r: %s", strategy, case.question, exc)
                records = []

            case_flags = [
                is_relevant_hit(case, record.chunk_id, record.text) for record in records
            ]
            hits_per_case.append(case_flags)
            totals_per_case.append(len(case.relevant_snippets) + len(case.relevant_chunk_ids))

            if generate:
                effective_top_k = top_k_override or self.settings.rerank_top_k
                final_records = records[:effective_top_k]
                citations = build_citations(final_records)
                draft = await self.rag.generate(query=case.question, citations=citations)
                verification = await self.verifier.verify(draft.answer, citations)
                tracker.observe(verification)

        metrics = RetrievalStrategyMetrics(
            strategy=strategy,
            num_cases=len(cases),
            recall_at_k={str(k): recall_at_k(hits_per_case, totals_per_case, k) for k in ks},
            mrr=reciprocal_rank(hits_per_case),
        )
        faithfulness = tracker.to_stats() if generate else None
        return StrategyReport(retrieval=metrics, faithfulness=faithfulness)


def build_citations(records: list, citation_offset: int = 1) -> list[SourceCitation]:
    return [
        SourceCitation(
            citation_id=index + citation_offset,
            chunk_id=record.chunk_id,
            source_doc=record.document_name,
            section_heading=record.section_heading,
            text=record.text,
            relevance_score=float(min(max(record.score, 0.0), 1.0)),
            page=record.page,
        )
        for index, record in enumerate(records)
    ]


def build_markdown_table(report_results: dict[str, StrategyReport], ks: list[int]) -> str:
    header_cells = ["Strategy", "Cases"] + [f"Recall@{k}" for k in ks] + ["MRR", "Groundedness"]
    rows = ["| " + " | ".join(header_cells) + " |", "|" + "---|" * len(header_cells)]
    for name in ("dense", "sparse", "hybrid_rrf", "hybrid_rerank"):
        report = report_results.get(name)
        if report is None:
            continue
        cells = [name.replace("_", " "), str(report.retrieval.num_cases)]
        cells += [f"{report.retrieval.recall_at_k.get(str(k), 0.0):.3f}" for k in ks]
        cells.append(f"{report.retrieval.mrr:.3f}")
        groundedness = report.faithfulness.groundedness_rate if report.faithfulness else None
        cells.append(f"{groundedness:.3f}" if groundedness is not None else "-")
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


async def evaluate_from_request(rag: RAGService, verifier: CitationVerifier, settings: Settings, request) -> tuple[EvalReport, Path, Path]:
    from app.schemas.eval import EvalRunRequest

    if not isinstance(request, EvalRunRequest):
        raise EvaluationError("Invalid evaluation request payload")
    runner = EvalRunner(rag=rag, verifier=verifier, settings=settings)
    dataset_path = resolve_dataset_path(settings, request.dataset_path)
    report = await runner.run(
        dataset_path=dataset_path,
        ks=request.ks,
        strategies=request.strategies,
        include_generation=request.include_generation,
        top_k=request.top_k,
        limit=request.limit or settings.eval_max_cases,
    )
    json_path, md_path = save_report(report, settings.reports_dir)
    return report, json_path, md_path


def resolve_dataset_path(settings: Settings, raw_path: str | None) -> Path:
    candidate = Path(raw_path) if raw_path else settings.benchmark_path
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
        if not candidate.exists():
            alternative = settings.data_dir.parent / candidate.name
            if alternative.exists():
                return alternative
    return candidate


def save_report(report: EvalReport, reports_dir: Path) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"eval_{report.run_id}.json"
    md_path = reports_dir / f"eval_{report.run_id}.md"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(render_markdown_report(report), encoding="utf-8")
    logger.info("Saved evaluation artifacts: %s, %s", json_path, md_path)
    return json_path, md_path


def render_markdown_report(report: EvalReport) -> str:
    lines = [
        "# RAG Benchmark Report",
        "",
        f"- **Run id**: `{report.run_id}`",
        f"- **Generated at**: {report.generated_at.isoformat()}",
        f"- **Dataset**: `{report.dataset}`",
        f"- **Cases evaluated**: {report.num_cases}",
        f"- **Duration**: {report.duration_seconds:.2f}s",
        "",
        report.markdown_table,
        "",
    ]
    for name, strategy_report in report.results.items():
        faithfulness = strategy_report.faithfulness
        if faithfulness is None:
            continue
        lines += [
            f"## Faithfulness — {name}",
            "",
            f"- Claims: {faithfulness.total_claims} "
            f"(supported={faithfulness.supported_claims}, "
            f"contradicted={faithfulness.contradicted_claims}, "
            f"unverified={faithfulness.unverified_claims})",
            f"- Groundedness rate: {faithfulness.groundedness_rate:.2%}",
            f"- Fully grounded answers: {faithfulness.fully_grounded_answers}/{faithfulness.total_answers} "
            f"({faithfulness.answer_grounding_rate:.2%})",
            "",
        ]
    return "\n".join(lines)
