"""Typed exception hierarchy mapped to HTTP responses in the API layer."""


class RAGError(Exception):
    """Base class for all domain errors raised by the engine."""

    http_status: int = 500
    error_code: str = "rag_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ParsingError(RAGError):
    http_status = 422
    error_code = "document_parse_failed"


class IngestionError(RAGError):
    http_status = 500
    error_code = "ingestion_failed"


class RetrievalError(RAGError):
    http_status = 503
    error_code = "retrieval_failed"


class GenerationError(RAGError):
    http_status = 502
    error_code = "generation_failed"


class EvaluationError(RAGError):
    http_status = 400
    error_code = "evaluation_failed"
