"""
Flask service — synchronous request handling, served in production by
Gunicorn's default sync workers (see ../Dockerfile.flask). Deliberately
mirrors the FastAPI service's endpoints and response shape exactly, so the
two are a fair, controlled comparison of serving stack rather than of two
different APIs.
"""

import logging
import os
import sys
import time

from flask import Flask, jsonify, request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.search_engine import matcher, MODEL_VERSION  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='{"ts": "%(asctime)s", "level": "%(levelname)s", "service": "flask", "msg": "%(message)s"}',
)
logger = logging.getLogger("job-fit-matcher.flask")

START_TIME = time.time()
REQUEST_COUNT = {"total": 0, "errors": 0}


def create_app() -> Flask:
    app = Flask(__name__)

    load_seconds = matcher.load()
    logger.info(f"Model loaded in {load_seconds:.3f}s, version={MODEL_VERSION}")

    @app.get("/health")
    def health():
        return jsonify({
            "status": "ok" if matcher.is_ready else "loading",
            "service": "flask",
            "model_version": MODEL_VERSION,
            "uptime_seconds": round(time.time() - START_TIME, 1),
            "requests_served": REQUEST_COUNT["total"],
        })

    @app.get("/metrics")
    def metrics():
        return jsonify({
            "requests_total": REQUEST_COUNT["total"],
            "requests_errors": REQUEST_COUNT["errors"],
            "uptime_seconds": round(time.time() - START_TIME, 1),
        })

    @app.post("/match")
    def match():
        REQUEST_COUNT["total"] += 1
        body = request.get_json(silent=True) or {}
        resume_text = body.get("resume_text", "")
        top_k = int(body.get("top_k", 5))

        if len(resume_text) < 10:
            REQUEST_COUNT["errors"] += 1
            return jsonify({"detail": "resume_text must be at least 10 characters"}), 422
        if not (1 <= top_k <= 16):
            REQUEST_COUNT["errors"] += 1
            return jsonify({"detail": "top_k must be between 1 and 16"}), 422
        if not matcher.is_ready:
            REQUEST_COUNT["errors"] += 1
            return jsonify({"detail": "Model not loaded yet"}), 503

        t0 = time.time()
        try:
            results = matcher.rank_postings(resume_text, top_k=top_k)
        except Exception as exc:
            REQUEST_COUNT["errors"] += 1
            logger.exception("Inference failed")
            return jsonify({"detail": str(exc)}), 500

        inference_ms = (time.time() - t0) * 1000
        logger.info(f"match served in {inference_ms:.2f}ms, top_k={top_k}")

        return jsonify({
            "model_version": MODEL_VERSION,
            "query_length_chars": len(resume_text),
            "inference_ms": round(inference_ms, 2),
            "results": results,
        })

    return app


app = create_app()
