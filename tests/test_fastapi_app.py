"""Unit tests for the FastAPI service, using FastAPI's TestClient (no
network, no running server needed)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from app_fastapi.main import app


@pytest.fixture(scope="module")
def client():
    # Using TestClient as a context manager triggers the app's lifespan
    # (startup/shutdown) events, which is what actually loads the model —
    # without it, matcher.is_ready stays False regardless of import order.
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "fastapi"


def test_match_returns_ranked_results(client):
    resp = client.post("/match", json={
        "resume_text": "Experienced Python backend engineer with Docker and FastAPI production deployment experience.",
        "top_k": 3,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 3
    assert body["results"][0]["rank"] == 1
    # results should be sorted descending by score
    scores = [r["score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)


def test_match_rejects_short_resume_text(client):
    resp = client.post("/match", json={"resume_text": "short"})
    assert resp.status_code == 422


def test_match_rejects_bad_top_k(client):
    resp = client.post("/match", json={
        "resume_text": "A valid resume text that is long enough to pass validation.",
        "top_k": 0,
    })
    assert resp.status_code == 422


def test_metrics_endpoint_reports_counts(client):
    before = client.get("/metrics").json()["requests_total"]
    client.post("/match", json={"resume_text": "Another valid resume text for testing purposes here."})
    after = client.get("/metrics").json()["requests_total"]
    assert after == before + 1
