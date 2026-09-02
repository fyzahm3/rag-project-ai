"""Offline end-to-end tests: chunkers, RRF, metrics, verifier, ingestion dedup, full pipeline, API."""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.evaluation.dataset import is_relevant_hit, load_cases
from app.evaluation.metrics import EvalRunner, recall_at_k, reciprocal_rank
from app.generation.generator import INSUFFICIENT_PREFIX, GroundedGenerator
from app.generation.verifier import CitationVerifier, extract_claims
from app.ingestion.chunker import (
    FixedChunker,
    SemanticBoundaryChunker,
    StructureAwareChunker,
    get_chunker,
)
from app.ingestion.indexer import Indexer
from app.ingestion.parser import ParsedDocument, ParsedSection, parse_document
from app.pipeline import RAGService
from app.retrieval.dense import DenseRetriever, NumpyVectorStore
from app.retrieval.embeddings import HashingEmbedder
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.sparse import SparseIndex
from app.schemas.eval import StrategyName
from app.schemas.query import QueryRequest, SourceCitation


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        raw_docs_dir=tmp_path / "raw_docs",
        chroma_dir=tmp_path / "chroma",
        index_dir=tmp_path / "index",
        benchmark_path=tmp_path / "benchmarks" / "golden_qa.jsonl",
        reports_dir=tmp_path / "reports",
        vector_store_backend="numpy",
        embedding_provider="huggingface",
        warm_models=False,
        verifier_mode="heuristic",
        min_chunk_chars=40,
        log_level="WARNING",
    )


@pytest.fixture()
def components(settings: Settings):
    embedder = HashingEmbedder(dim=128)
    store = NumpyVectorStore()
    sparse = SparseIndex(settings.index_dir / "bm25_corpus.jsonl")
    dense = DenseRetriever(embedder=embedder, store=store)
    indexer = Indexer(settings=settings, embedder=embedder, vector_store=store, sparse_index=sparse)
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
    return {"settings": settings, "embedder": embedder, "store": store, "sparse": sparse,
            "dense": dense, "indexer": indexer, "reranker": reranker,
            "generator": generator, "verifier": verifier, "rag": rag}


SAMPLE_MD = """# Platform Guide

## Storage Layer

All stateful workloads use the Ceph-backed storage class ceph-rbd-ssd.
The storage class provisions volumes in under 900 milliseconds with replication factor three.

## Networking

Ingress traffic terminates at Envoy proxies that enforce mutual TLS for east-west requests.
Network policies default-deny cross namespace traffic unless explicitly allowed."""


def test_fixed_chunker_window_and_overlap(settings: Settings):
    doc = ParsedDocument(document_name="flat.txt", doc_type="text", sections=[
        ParsedSection(heading="body", text=" ".join(["alpha beta gamma delta"] * 60)),
    ])
    settings.chunk_size = 300
    settings.chunk_overlap = 60
    chunks = FixedChunker(settings).chunk(doc)
    assert len(chunks) >= 3
    assert all(len(c.text) <= 320 for c in chunks)
    step = 300 - 60
    assert chunks[1].seq == 1


def test_structure_aware_chunker_preserves_sections(settings: Settings):
    doc = parse_document("guide.md", SAMPLE_MD.encode("utf-8"))
    assert doc.doc_type == "markdown"
    chunks = StructureAwareChunker(settings).chunk(doc)
    headings = {c.section_heading for c in chunks}
    assert "Storage Layer" in headings
    assert "Networking" in headings


def test_semantic_chunker_splits_on_topic_shift(settings: Settings):
    fruit = ["Apples are a sweet fruit grown worldwide.",
             "Bananas contain potassium and natural sugars."]
    database = ["Postgres tables store relational rows on disk.",
                "Indexes accelerate primary key lookups in databases."]
    sentences = fruit + database

    def fake_embed(texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            if text in set(fruit):
                base = np.array([1.0, 0.05])
            else:
                base = np.array([0.05, 1.0])
            vectors.append(base + np.random.RandomState(len(text)).normal(0, 0.001, 2))
        return np.stack(vectors)

    settings.semantic_distance_threshold = 0.4
    settings.chunk_size = 100
    doc = ParsedDocument(document_name="mixed.txt", doc_type="text", sections=[
        ParsedSection(heading="topics", text=" ".join(sentences)),
    ])
    chunks = SemanticBoundaryChunker(settings, embed_fn=fake_embed).chunk(doc)
    assert len(chunks) >= 2
    joined = " ".join(c.text for c in chunks)
    assert "Apples" in joined and "Postgres" in joined


def test_rrf_fusion_order_and_weights():
    from app.retrieval.dense import RetrievedRecord

    def rec(cid: str) -> RetrievedRecord:
        return RetrievedRecord(chunk_id=cid, text=f"text-{cid}", document_name="d")

    dense_list = [rec("A"), rec("B"), rec("C")]
    sparse_list = [rec("C"), rec("A"), rec("D")]

    fused_equal = reciprocal_rank_fusion([dense_list, sparse_list], k=60)
    ids_equal = [r.chunk_id for r in fused_equal]
    assert ids_equal[0] == "A"
    assert ids_equal[1] == "C"
    assert fused_equal[0].raw_score > fused_equal[1].raw_score
    assert set(ids_equal) == {"A", "B", "C", "D"}

    fused_weighted = reciprocal_rank_fusion(
        [dense_list, sparse_list], k=60, weights=[10.0, 0.1]
    )
    ids_weighted = [r.chunk_id for r in fused_weighted]
    assert ids_weighted[0] == "A"
    assert fused_weighted[1].chunk_id == "B"
    assert 0 < fused_weighted[0].score <= 1.0


def test_recall_and_mrr_math():
    hits = [[True, False, True], [False, True], [False, False]]
    totals = [2, 2, 2]
    assert recall_at_k(hits, totals, k=1) == pytest.approx((0.5 + 0.0 + 0.0) / 3)
    assert recall_at_k(hits, totals, k=2) == pytest.approx((0.5 + 0.5 + 0.0) / 3)
    rr = reciprocal_rank(hits)
    assert rr == pytest.approx((1.0 + 0.5 + 0.0) / 3)


def test_citation_extraction_and_heuristic_verification(components):
    citations = [
        SourceCitation(citation_id=1, chunk_id="c1", source_doc="doc.md", section_heading="S",
                       text="The storage class ceph-rbd-ssd provisions volumes in under 900 milliseconds.",
                       relevance_score=0.9),
        SourceCitation(citation_id=2, chunk_id="c2", source_doc="doc.md", section_heading="N",
                       text="Ingress traffic terminates at Envoy proxies enforcing mutual TLS.",
                       relevance_score=0.8),
    ]
    answer = ("Volumes provision in under 900 milliseconds [1]. "
              "The gateway uses Envoy proxies for ingress [2]. "
              "Quantum encryption is enabled by default [2].")
    claims = extract_claims(answer)
    assert [len(claim.citation_ids) for claim in claims] == [1, 1, 1]

    verifier: CitationVerifier = components["verifier"]
    verification = asyncio.run(verifier.verify(answer, citations))
    assert verification["engine"] == "heuristic"
    statuses = [claim["status"] for claim in verification["claims"]]
    assert "SUPPORTED" in statuses
    assert verification["counts"]["CONTRADICTED"] == 0
    assert 0.0 <= verification["pass_rate"] <= 1.0


def test_insufficient_context_short_circuit(components):
    verifier: CitationVerifier = components["verifier"]
    verification = asyncio.run(verifier.verify(f"{INSUFFICIENT_PREFIX}: nothing relevant.", []))
    assert verification["flag"] == "insufficient_context"
    assert verification["claims"] == []


def test_extractive_generator_cites_sources(components):
    generator: GroundedGenerator = components["generator"]
    citations = [
        SourceCitation(citation_id=1, chunk_id="c1", source_doc="doc.md", section_heading="S",
                       text="Backups run hourly and are retained for thirty days in object storage.",
                       relevance_score=0.9),
    ]
    draft = asyncio.run(generator.generate("How long are backups retained?", citations))
    assert "[1]" in draft.answer
    assert draft.llm_used is False
    empty = asyncio.run(generator.generate("unknown topic?", []))
    assert empty.answer.startswith(INSUFFICIENT_PREFIX)


def test_ingest_deduplicates_on_second_pass(components):
    indexer: Indexer = components["indexer"]
    first = asyncio.run(indexer.ingest("guide.md", SAMPLE_MD.encode("utf-8")))
    assert first.chunks_created >= 2
    second = asyncio.run(indexer.ingest("guide.md", SAMPLE_MD.encode("utf-8")))
    assert second.chunks_created == 0
    assert second.deduplicated_chunks >= first.chunks_created


def test_full_offline_pipeline_ask(components, settings: Settings):
    indexer: Indexer = components["indexer"]
    rag: RAGService = components["rag"]
    asyncio.run(indexer.ingest("guide.md", SAMPLE_MD.encode("utf-8")))

    response = asyncio.run(rag.ask(QueryRequest(
        query="How fast are volumes provisioned?",
        top_k=3,
        enable_reranking=False,
    )))
    assert response.citations, "expected at least one citation"
    assert response.retrieval_latency_ms >= 0
    assert response.generation_latency_ms >= 0
    assert 0.0 <= response.confidence_score <= 1.0
    assert response.verification_status["engine"] in {"heuristic", "nli_cross_encoder", "llm_judge"}

    miss = asyncio.run(rag.ask(QueryRequest(query="What is the quarterly revenue of Unobtainium Corp?")))
    assert miss.answer.startswith(INSUFFICIENT_PREFIX) or miss.confidence_score < 0.6


def test_eval_runner_dense_strategy(tmp_path: Path, components):
    indexer: Indexer = components["indexer"]
    rag: RAGService = components["rag"]
    settings: Settings = components["settings"]
    asyncio.run(indexer.ingest("guide.md", SAMPLE_MD.encode("utf-8")))

    dataset_path = tmp_path / "golden.jsonl"
    dataset_path.write_text(
        '{"question": "How fast are volumes provisioned?", '
        '"relevant_snippets": ["provisions volumes in under 900 milliseconds"]}\n'
        '{"question": "Who won the 1904 interplanetary world cup?", '
        '"relevant_snippets": ["interplanetary world cup trophy"]}\n',
        encoding="utf-8",
    )
    cases = load_cases(dataset_path)
    assert len(cases) == 2
    assert is_relevant_hit(
        cases[0], "whatever",
        "The storage class provisions volumes in under 900 milliseconds with replication factor three.",
    )
    assert not is_relevant_hit(cases[0], "other", "Completely unrelated content about weather.")

    runner = EvalRunner(rag=rag, verifier=components["verifier"], settings=settings)
    report = asyncio.run(runner.run(dataset_path=dataset_path, ks=[1, 3],
                                    strategies=["dense"], include_generation=False))
    dense_report = report.results["dense"]
    assert dense_report.retrieval.num_cases == 2
    assert dense_report.faithfulness is None
    assert report.markdown_table.count("|") > 4


def test_api_endpoints_with_fake_context(components):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from app.main import AppContext, create_app

    ctx = AppContext(
        settings=components["settings"],
        embedder=components["embedder"],
        store=components["store"],
        sparse=components["sparse"],
        dense=components["dense"],
        indexer=components["indexer"],
        reranker=components["reranker"],
        generator=components["generator"],
        verifier=components["verifier"],
        rag=components["rag"],
    )
    app = create_app(ctx)
    with fastapi_testclient.TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        body = health.json()
        assert body["index"]["vector_store"] == "numpy"
        assert body["models"]["llm_available"] is False

        ingest = client.post(
            "/v1/ingest",
            files={"file": ("guide.md", SAMPLE_MD.encode("utf-8"), "text/markdown")},
            data={"strategy": "structure"},
        )
        assert ingest.status_code == 200
        assert ingest.json()["chunks_created"] >= 2

        ask = client.post("/v1/ask", json={"query": "How fast are volumes provisioned?", "top_k": 3})
        assert ask.status_code == 200
        ask_body = ask.json()
        assert ask_body["citations"]
        assert "verification_status" in ask_body

        bad_upload = client.post(
            "/v1/ingest",
            files={"file": ("malware.exe", b"MZ...", "application/octet-stream")},
        )
        assert bad_upload.status_code == 415


def test_get_chunker_factory_validation(settings: Settings):
    with pytest.raises(ValueError):
        get_chunker("bogus-strategy", settings)
