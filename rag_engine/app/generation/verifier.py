"""Automated citation verification: parse [N] claims and classify entailment per cited chunk."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Literal

from pydantic import BaseModel

from app.config import Settings
from app.generation.generator import INSUFFICIENT_PREFIX
from app.schemas.query import SourceCitation
from app.utils.text import split_sentences, token_coverage, tokenize, truncate

logger = logging.getLogger(__name__)

CITATION_PATTERN = re.compile(r"\[(\d{1,3})\]")
VerificationLabel = Literal["SUPPORTED", "CONTRADICTED", "UNVERIFIED"]
VerifierEngine = Literal["llm_judge", "nli_cross_encoder", "heuristic"]

_NLI_LABELS = ["contradiction", "entailment", "neutral"]


class ParsedClaim(BaseModel):
    sentence: str
    citation_ids: list[int]


class ClaimVerificationResult(ParsedClaim):
    status: VerificationLabel = "UNVERIFIED"
    rationale: str = ""


def extract_claims(answer: str) -> list[ParsedClaim]:
    """Split the answer into sentences and attach every bracketed citation id found in each."""
    claims: list[ParsedClaim] = []
    for sentence in split_sentences(answer):
        ids = [int(match) for match in CITATION_PATTERN.findall(sentence)]
        deduped = sorted(dict.fromkeys(ids))
        if deduped:
            claims.append(ParsedClaim(sentence=sentence.strip(), citation_ids=deduped))
    return claims


class EntailmentJudge:
    engine_name: VerifierEngine = "heuristic"

    async def judge(self, claim: str, evidence: str) -> tuple[VerificationLabel, str]:
        raise NotImplementedError


class HeuristicJudge(EntailmentJudge):
    """Lexical-overlap entailment proxy; never asserts CONTRADICTED."""

    engine_name = "heuristic"

    def __init__(self, settings: Settings) -> None:
        self._supported_threshold = settings.entailment_supported_threshold
        self._unverified_threshold = settings.entailment_unverified_threshold

    async def judge(self, claim: str, evidence: str) -> tuple[VerificationLabel, str]:
        coverage = token_coverage(tokenize(claim), tokenize(evidence))
        if coverage >= self._supported_threshold:
            return "SUPPORTED", f"lexical coverage {coverage:.2f}"
        if coverage >= self._unverified_threshold:
            return "UNVERIFIED", f"partial lexical coverage {coverage:.2f}"
        return "UNVERIFIED", f"insufficient lexical overlap {coverage:.2f}"


class NLICrossEncoderJudge(EntailmentJudge):
    """Local NLI cross-encoder (premise=evidence, hypothesis=claim)."""

    engine_name = "nli_cross_encoder"

    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.nli_model_name
        self._model = None

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        try:
            from sentence_transformers import CrossEncoder

            logger.info("Loading NLI model '%s'", self._model_name)
            self._model = CrossEncoder(self._model_name, max_length=512)
            return True
        except Exception as exc:
            logger.warning("NLI model '%s' unavailable (%s)", self._model_name, exc)
            return False

    async def judge(self, claim: str, evidence: str) -> tuple[VerificationLabel, str]:
        import numpy as np

        if not self._ensure_model():
            raise RuntimeError(f"NLI model {self._model_name} could not be loaded")
        scores = await asyncio.to_thread(
            self._model.predict, [[evidence[:1500], claim[:500]]]
        )
        probabilities = _softmax(np.asarray(scores, dtype=np.float64).ravel())
        contradiction_p, entailment_p, neutral_p = (
            float(probabilities[0]),
            float(probabilities[1]),
            float(probabilities[2]),
        )
        if entailment_p >= 0.55:
            return "SUPPORTED", f"p(entail)={entailment_p:.2f} contra={contradiction_p:.2f} neutral={neutral_p:.2f}"
        if contradiction_p >= 0.55:
            return "CONTRADICTED", f"p(contra)={contradiction_p:.2f} entail={entailment_p:.2f}"
        return "UNVERIFIED", f"p(entail)={entailment_p:.2f} p(contra)={contradiction_p:.2f}"

    @property
    def available(self) -> bool:
        return self._ensure_model()


class LLMEntailmentJudge(EntailmentJudge):
    """LLM-as-judge with strict JSON output."""

    engine_name = "llm_judge"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            if not self._settings.openai_api_key:
                raise RuntimeError("LLM judge requested without OPENAI_API_KEY")
            self._client = AsyncOpenAI(
                api_key=self._settings.openai_api_key,
                timeout=self._settings.openai_timeout_seconds,
                max_retries=self._settings.openai_max_retries,
            )
        return self._client

    async def judge(self, claim: str, evidence: str) -> tuple[VerificationLabel, str]:
        client = self._ensure_client()
        prompt = (
            "You are a strict entailment judge. Decide whether the EVIDENCE strictly entails "
            "the CLAIM. Allowed labels: SUPPORTED, CONTRADICTED, UNVERIFIED.\n\n"
            f"EVIDENCE:\n{truncate(evidence, 1500)}\n\n"
            f"CLAIM:\n{truncate(claim, 600)}\n\n"
            'Reply with JSON only: {"label": "...", "rationale": "one short sentence"}'
        )
        response = await client.chat.completions.create(
            model=self._settings.openai_chat_model,
            temperature=0.0,
            max_tokens=120,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(content)
            label = str(parsed.get("label", "UNVERIFIED")).upper()
            if label not in {"SUPPORTED", "CONTRADICTED", "UNVERIFIED"}:
                label = "UNVERIFIED"
            return label, str(parsed.get("rationale", ""))[:200]
        except json.JSONDecodeError:
            return "UNVERIFIED", "judge returned malformed JSON"


def _softmax(values: np.ndarray) -> np.ndarray:
    import numpy as np

    shifted = values - values.max()
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()


class CitationVerifier:
    """Parses citations, judges each claim against its cited chunks, and aggregates confidence."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.mode = settings.verifier_mode
        self._judge: EntailmentJudge | None = None
        self._engine_used: VerifierEngine = "heuristic"
        self._nli_candidate = NLICrossEncoderJudge(settings)
        self._llm_candidate = LLMEntailmentJudge(settings)
        self._heuristic = HeuristicJudge(settings)

    @property
    def engine(self) -> VerifierEngine:
        return self._engine_used

    def _resolve_judge(self) -> EntailmentJudge:
        if self._judge is not None:
            return self._judge
        mode = self.mode
        if mode == "auto":
            mode = "llm" if self.settings.openai_api_key else "nli"
        if mode == "llm":
            try:
                self._llm_candidate._ensure_client()
                self._judge = self._llm_candidate
                self._engine_used = "llm_judge"
                return self._judge
            except Exception as exc:
                logger.warning("LLM judge unavailable (%s); falling back", exc)
                mode = "nli"
        if mode == "nli":
            if self._nli_candidate.available:
                self._judge = self._nli_candidate
                self._engine_used = "nli_cross_encoder"
                return self._judge
            logger.warning("NLI judge unavailable; using heuristic entailment")
        self._judge = self._heuristic
        self._engine_used = "heuristic"
        return self._judge

    async def verify(self, answer: str, citations: list[SourceCitation]) -> dict:
        counts = {"SUPPORTED": 0, "CONTRADICTED": 0, "UNVERIFIED": 0}
        flag = "ok"

        if answer.startswith(INSUFFICIENT_PREFIX):
            return {
                "engine": self.engine,
                "flag": "insufficient_context",
                "counts": counts,
                "pass_rate": 0.0,
                "claims": [],
            }

        claims = extract_claims(answer)
        if not claims:
            return {
                "engine": self.engine,
                "flag": flag,
                "counts": counts,
                "pass_rate": 0.0,
                "claims": [],
            }

        citation_map = {c.citation_id: c for c in citations}
        results = await asyncio.gather(*[self._verify_claim(claim, citation_map) for claim in claims])

        claim_payloads: list[dict] = []
        for result in results:
            counts[result.status] += 1
            claim_payloads.append(result.model_dump())

        total = len(results)
        pass_rate = counts["SUPPORTED"] / total if total else 0.0
        return {
            "engine": self.engine,
            "flag": flag,
            "counts": counts,
            "pass_rate": round(pass_rate, 4),
            "claims": claim_payloads,
        }

    async def _verify_claim(
        self,
        claim: ParsedClaim,
        citation_map: dict[int, SourceCitation],
    ) -> ClaimVerificationResult:
        judge = self._resolve_judge()
        statuses: list[VerificationLabel] = []
        rationales: list[str] = []
        valid_ids = [cid for cid in claim.citation_ids if cid in citation_map]

        if not valid_ids:
            return ClaimVerificationResult(
                sentence=claim.sentence,
                citation_ids=claim.citation_ids,
                status="UNVERIFIED",
                rationale="citation id not present in retrieved context set",
            )

        semaphore = asyncio.Semaphore(4)

        async def judged(cid: int) -> tuple[VerificationLabel, str]:
            async with semaphore:
                evidence = truncate(citation_map[cid].text, self.settings.evidence_max_chars)
                try:
                    return await judge.judge(claim.sentence, evidence)
                except Exception as exc:
                    logger.debug("Judge error on citation %d: %s", cid, exc)
                    heuristic_label, heuristic_rationale = await self._heuristic.judge(
                        claim.sentence, evidence
                    )
                    return heuristic_label, f"fallback:{heuristic_rationale}"

        outcomes = await asyncio.gather(*[judged(cid) for cid in valid_ids])
        statuses = [label for label, _ in outcomes]
        rationales = [f"[{cid}] {rationale}" for cid, (_, rationale) in zip(valid_ids, outcomes)]

        if "SUPPORTED" in statuses:
            final_status: VerificationLabel = "SUPPORTED"
        elif all(status == "CONTRADICTED" for status in statuses):
            final_status = "CONTRADICTED"
        elif "CONTRADICTED" in statuses:
            final_status = "CONTRADICTED"
        else:
            final_status = "UNVERIFIED"

        return ClaimVerificationResult(
            sentence=claim.sentence,
            citation_ids=valid_ids,
            status=final_status,
            rationale="; ".join(rationales)[:400],
        )


def compute_confidence(
    mean_relevance: float,
    verification: dict,
    retrieval_weight: float,
    verification_weight: float,
    contradiction_penalty: float,
) -> float:
    """Composite score: retrieval similarity blended with citation verification pass rate."""
    counts = verification.get("counts", {})
    total_claims = sum(counts.values())
    pass_rate = verification.get("pass_rate", 0.0)
    if verification.get("flag") == "insufficient_context" or total_claims == 0:
        base = retrieval_weight * mean_relevance + verification_weight * 0.0
        return round(max(0.0, min(base, 1.0)), 4)

    contradicted_ratio = counts.get("CONTRADICTED", 0) / total_claims
    confidence = (
        retrieval_weight * mean_relevance
        + verification_weight * pass_rate
        - contradiction_penalty * contradicted_ratio
    )
    return round(max(0.0, min(confidence, 1.0)), 4)
