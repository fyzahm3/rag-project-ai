"""Seed the engine with sample technical docs and produce an initial evaluation report.

Usage:
    python scripts/seed_data.py            # seed if empty + evaluate
    python scripts/seed_data.py --force    # re-ingest even if already seeded
    python scripts/seed_data.py --skip-eval
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.evaluation.metrics import EvalRunner, render_markdown_report, save_report
from app.generation.verifier import CitationVerifier
from app.main import build_context, configure_logging

MARKDOWN_DOC = """# Container Platform Standards

## Runtime Architecture

The production platform runs Kubernetes 1.29 across three availability zones with 240 worker nodes.
Every node runs Talos Linux with an immutable root filesystem, and kubelet versions may lag the control plane by at most one minor release.

Cluster autoscaler maintains a minimum of twelve nodes per zone and scales to forty nodes under sustained peak load.
Node pools are labeled by workload class so that latency-sensitive services land on dedicated compute.

## Storage Layer

All stateful workloads use the Ceph-backed storage class `ceph-rbd-ssd`, which provisions volumes in under 900 milliseconds.
Replication factor three guarantees durability, and volume snapshots are taken every six hours.

Database backups run hourly and are retained for thirty days in object storage.
Restore drills execute monthly against a staging cluster to validate recovery time objectives.

## Networking

Ingress traffic terminates at Envoy proxies that enforce mutual TLS for every east-west request.
Network policies default-deny all cross-namespace traffic unless explicitly allowed by the platform security baseline.

## Upgrade Policy

Control plane upgrades follow a rolling schedule every ninety days, with node pools drained one zone at a time.
A canary fleet receives each release seven days before general availability."""

TEXT_DOC = """API GATEWAY REFERENCE

RATE LIMITING:
Public endpoints allow 120 requests per minute per API key.
Bursts up to 300 requests are absorbed by the token bucket for ten seconds before throttling returns HTTP 429.

AUTHENTICATION:
All clients authenticate with OAuth 2.0 client credentials.
Access tokens expire after fifteen minutes and refresh tokens after eight hours.
Service accounts must rotate client secrets at least once per quarter.

CACHING:
GET responses are cached for sixty seconds at the edge.
Cache invalidation propagates within five seconds via pub-sub fanout to all points of presence.

ERROR SEMANTICS:
The gateway maps upstream 5xx responses to a stable error envelope with a trace id.
Clients should retry idempotent calls with exponential backoff and a maximum of three attempts."""

HTML_DOC = """<!DOCTYPE html>
<html>
<head><title>Observability Stack</title></head>
<body>
<h1>Observability Stack</h1>
<h2>Metrics Pipeline</h2>
<p>Prometheus scrapes every pod at a fifteen second interval.</p>
<p>Remote-write shards metrics to Thanos, giving the team thirteen months of retention for capacity planning.</p>
<h2>Distributed Tracing</h2>
<p>Traces sample ten percent of requests and propagate W3C traceparent headers across all services.</p>
<p>The tracing backend keeps high-cardinality spans for thirty days before downsampling.</p>
<h2>Alerting</h2>
<p>Pages fire when the p99 latency exceeds four hundred milliseconds for five consecutive minutes.</p>
<p>All alerts route through the on-call rotation with a two-stage escalation policy.</p>
</body>
</html>"""

PDF_PAGES: list[tuple[str, list[str]]] = [
    (
        "Data Architecture Overview",
        [
            "The lakehouse stores raw events in Delta tables partitioned by ingestion date.",
            "Stream ingestion through Kafka achieves end-to-end latency under two seconds from producer to queryable table.",
        ],
    ),
    (
        "Quality Enforcement",
        [
            "Every pipeline run executes Great Expectations suites before promotion.",
            "Quarantined records exceeding one percent abort the deployment automatically and page the data-on-call engineer.",
        ],
    ),
]

GOLDEN_CASES = [
    {
        "question": "How fast are volumes provisioned on the platform storage layer?",
        "expected_answer": "Volumes on the ceph-rbd-ssd class provision in under 900 milliseconds.",
        "relevant_snippets": ["provisions volumes in under 900 milliseconds"],
        "tags": ["storage"],
    },
    {
        "question": "What is the public API rate limit per key?",
        "expected_answer": "120 requests per minute per API key.",
        "relevant_snippets": ["allow 120 requests per minute per API key"],
        "tags": ["api-gateway"],
    },
    {
        "question": "What fraction of requests are sampled for distributed tracing?",
        "expected_answer": "Ten percent of requests are sampled.",
        "relevant_snippets": ["Traces sample ten percent of requests"],
        "tags": ["observability"],
    },
    {
        "question": "How long are database backups retained?",
        "expected_answer": "Backups are retained for thirty days in object storage.",
        "relevant_snippets": ["retained for thirty days in object storage"],
        "tags": ["storage"],
    },
    {
        "question": "What end-to-end latency does Kafka stream ingestion achieve?",
        "expected_answer": "Under two seconds from producer to queryable table.",
        "relevant_snippets": ["end-to-end latency under two seconds"],
        "tags": ["data"],
    },
    {
        "question": "When does alerting page the on-call engineer for latency?",
        "expected_answer": "When p99 latency exceeds four hundred milliseconds for five consecutive minutes.",
        "relevant_snippets": ["exceeds four hundred milliseconds for five consecutive minutes"],
        "tags": ["observability"],
    },
]


def write_pdf(path: Path) -> None:
    import fitz

    document = fitz.open()
    for heading, paragraphs in PDF_PAGES:
        page = document.new_page()
        page.insert_text((72, 90), heading, fontsize=18, fontname="helv")
        y = 130
        for paragraph in paragraphs:
            rect = fitz.Rect(72, y, page.rect.width - 72, page.rect.height - 72)
            leftover = page.insert_textbox(rect, paragraph, fontsize=11, fontname="helv")
            if leftover < 0:
                page.insert_text((72, y), paragraph[:400], fontsize=10, fontname="helv")
            y += 120
    document.save(str(path))
    document.close()


def write_seed_docs(raw_docs_dir: Path) -> list[Path]:
    raw_docs_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    md_path = raw_docs_dir / "kubernetes_platform.md"
    md_path.write_text(MARKDOWN_DOC, encoding="utf-8")
    written.append(md_path)

    txt_path = raw_docs_dir / "api_gateway.txt"
    txt_path.write_text(TEXT_DOC, encoding="utf-8")
    written.append(txt_path)

    html_path = raw_docs_dir / "observability.html"
    html_path.write_text(HTML_DOC, encoding="utf-8")
    written.append(html_path)

    try:
        pdf_path = raw_docs_dir / "data_architecture.pdf"
        write_pdf(pdf_path)
        written.append(pdf_path)
    except Exception as exc:
        logging.getLogger("seed").warning("Skipped PDF seed (PyMuPDF unavailable): %s", exc)
    return written


def write_golden_dataset(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for case in GOLDEN_CASES:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")


async def run_seed(force: bool, skip_eval: bool) -> int:
    settings = get_settings()
    configure_logging(settings)
    logger = logging.getLogger("seed")
    marker = settings.index_dir / ".seeded"

    context = build_context(settings)
    if marker.exists() and not force:
        logger.info("Index already seeded (%s exists); use --force to re-ingest", marker)
    else:
        docs = write_seed_docs(settings.raw_docs_dir)
        for doc_path in docs:
            response = await context.indexer.ingest(
                doc_path.name,
                doc_path.read_bytes(),
                strategy=settings.default_chunking_strategy,
            )
            logger.info(
                "Seeded %s -> chunks=%d deduplicated=%d",
                response.document_name,
                response.chunks_created,
                response.deduplicated_chunks,
            )
        marker.write_text("seeded\n", encoding="utf-8")

    if skip_eval:
        return 0

    if not settings.benchmark_path.exists():
        write_golden_dataset(settings.benchmark_path)

    runner = EvalRunner(rag=context.rag, verifier=context.verifier, settings=settings)
    report = await runner.run(include_generation=True)
    json_path, md_path = save_report(report, settings.reports_dir)

    print("\n" + report.markdown_table + "\n")
    print(f"Artifacts:\n  - {json_path}\n  - {md_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed sample docs and run initial evaluation")
    parser.add_argument("--force", action="store_true", help="Re-ingest even if previously seeded")
    parser.add_argument(
        "--if-empty",
        action="store_true",
        help="No-op alias: seeding already skips when the index marker exists",
    )
    parser.add_argument("--skip-eval", action="store_true", help="Skip the evaluation harness")
    args = parser.parse_args()
    return asyncio.run(run_seed(force=args.force, skip_eval=args.skip_eval))


if __name__ == "__main__":
    raise SystemExit(main())
