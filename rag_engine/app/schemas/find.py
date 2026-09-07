"""Schemas for GET /v1/find and POST /v1/open (fast file search, no rerank/LLM)."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FindHit(BaseModel):
    path: str
    snippet: str
    score: float = Field(ge=0.0, le=1.0)
    mtime: float | None = None


class FindResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "readme launch window",
                "results": [
                    {
                        "path": "/Users/you/Documents/readme.md",
                        "snippet": "The launch window opens at 0900 UTC and closes at 1100 UTC.",
                        "score": 0.98,
                        "mtime": 1730000000.0,
                    }
                ],
                "latency_ms": 42.3,
            }
        }
    )

    query: str
    results: list[FindHit]
    latency_ms: float


class OpenFileRequest(BaseModel):
    path: str = Field(min_length=1)
