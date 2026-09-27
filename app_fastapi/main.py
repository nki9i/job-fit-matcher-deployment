"""
FastAPI service — async request handling, served in production by Gunicorn
using `uvicorn.workers.UvicornWorker` worker processes (see ../Dockerfile.fastapi).
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Make the sibling `common/` package importable whether this runs from the
# repo root (local dev: `uvicorn app_fastapi.main:app`) or from /app inside
# the container, which mirrors the same repo-root layout.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.search_engine import matcher, MODEL_VERSION  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='{"ts": "%(asctime)s", "level": "%(levelname)s", "service": "fastapi", "msg": "%(message)s"}',
)
logger = logging.getLogger("job-fit-matcher.fastapi")

START_TIME = time.time()
REQUEST_COUNT = {"total": 0, "errors": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_seconds = matcher.load()
    logger.info(f"Model loaded in {load_seconds:.3f}s, version={MODEL_VERSION}")
    yield
    logger.info("Shutting down FastAPI service")


app = FastAPI(title="Job Fit Matcher — FastAPI service", version=MODEL_VERSION, lifespan=lifespan)


class MatchRequest(BaseModel):
    resume_text: str = Field(..., min_length=10, description="Resume or bio text to match against job postings")
    top_k: int = Field(5, ge=1, le=16, description="Number of ranked postings to return")


class MatchResult(BaseModel):
    rank: int
    posting_id: str
    title: str
    company: str
    location: str
    score: float
    matched_keywords: list[str]


class MatchResponse(BaseModel):
    model_version: str
    query_length_chars: int
    inference_ms: float
    results: list[MatchResult]


@app.get("/health")
def health():
    return {
        "status": "ok" if matcher.is_ready else "loading",
        "service": "fastapi",
        "model_version": MODEL_VERSION,
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "requests_served": REQUEST_COUNT["total"],
    }


@app.get("/metrics")
def metrics():
    return {
        "requests_total": REQUEST_COUNT["total"],
        "requests_errors": REQUEST_COUNT["errors"],
        "uptime_seconds": round(time.time() - START_TIME, 1),
    }


@app.post("/match", response_model=MatchResponse)
def match(req: MatchRequest):
    REQUEST_COUNT["total"] += 1
    if not matcher.is_ready:
        REQUEST_COUNT["errors"] += 1
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    t0 = time.time()
    try:
        results = matcher.rank_postings(req.resume_text, top_k=req.top_k)
    except Exception as exc:
        REQUEST_COUNT["errors"] += 1
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    inference_ms = (time.time() - t0) * 1000
    logger.info(f"match served in {inference_ms:.2f}ms, top_k={req.top_k}")

    return MatchResponse(
        model_version=MODEL_VERSION,
        query_length_chars=len(req.resume_text),
        inference_ms=round(inference_ms, 2),
        results=results,
    )
