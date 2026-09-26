"""HTTP API for Hindi summarization.

    uvicorn textsumm.api:app --port 8000
    TEXTSUMM_CHECKPOINT=checkpoints/indicbart-hindi uvicorn textsumm.api:app --port 8000

The extractive methods are always available. The abstractive model is loaded
on the first request that asks for it (from TEXTSUMM_CHECKPOINT), so the
service starts instantly and runs on machines without torch installed.
"""

import os
import threading
import time
from enum import Enum
from typing import Annotated, Protocol

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from textsumm import __version__
from textsumm.baseline import lead_n, textrank

MAX_BATCH = 32
MAX_CHARS = 20_000


class Method(str, Enum):
    model = "model"
    lead = "lead"
    textrank = "textrank"


class SummarizeRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_BATCH)
    method: Method = Method.model
    num_sentences: int = Field(1, ge=1, le=10, description="sentences to extract (lead/textrank only)")

    @field_validator("texts")
    @classmethod
    def check_texts(cls, texts: list[str]) -> list[str]:
        for text in texts:
            if not text.strip():
                raise ValueError("texts must not be empty")
            if len(text) > MAX_CHARS:
                raise ValueError(f"each text must be at most {MAX_CHARS} characters")
        return texts


class SummarizeResponse(BaseModel):
    summaries: list[str]
    method: Method
    latency_ms: float


class BatchSummarizer(Protocol):
    def summarize(self, texts: list[str], batch_size: int = 8) -> list[str]: ...


class ModelRegistry:
    """Loads the abstractive model once, on first use, and shares it across requests."""

    def __init__(self, checkpoint: str | None):
        self.checkpoint = checkpoint
        self._model: BatchSummarizer | None = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def get(self) -> BatchSummarizer:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    if not self.checkpoint:
                        raise RuntimeError("no model configured; set TEXTSUMM_CHECKPOINT")
                    from textsumm.infer import Summarizer

                    self._model = Summarizer(self.checkpoint)
        return self._model


registry = ModelRegistry(os.environ.get("TEXTSUMM_CHECKPOINT"))


def get_registry() -> ModelRegistry:
    return registry


Registry = Annotated[ModelRegistry, Depends(get_registry)]


app = FastAPI(title="Hindi News Summarizer", version=__version__)


@app.get("/health")
def health(models: Registry):
    return {"status": "ok", "model_configured": bool(models.checkpoint), "model_loaded": models.loaded}


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest, models: Registry):
    start = time.perf_counter()
    if request.method is Method.model:
        try:
            summarizer = models.get()
        except (RuntimeError, ImportError, OSError) as exc:
            raise HTTPException(status_code=503, detail=f"model unavailable: {exc}") from exc
        summaries = summarizer.summarize(request.texts)
    else:
        extract = lead_n if request.method is Method.lead else textrank
        summaries = [extract(text, request.num_sentences) for text in request.texts]

    return SummarizeResponse(
        summaries=summaries,
        method=request.method,
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
    )
