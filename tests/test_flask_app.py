"""Unit tests for the Flask service, using Flask's built-in test client."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_flask.app import app as flask_app

flask_app.testing = True
client = flask_app.test_client()


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert body["service"] == "flask"


def test_match_returns_ranked_results():
    resp = client.post("/match", json={
        "resume_text": "Experienced Python backend engineer with Docker and FastAPI production deployment experience.",
        "top_k": 3,
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert len(body["results"]) == 3
    assert body["results"][0]["rank"] == 1
    scores = [r["score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)


def test_match_rejects_short_resume_text():
    resp = client.post("/match", json={"resume_text": "short"})
    assert resp.status_code == 422


def test_match_rejects_bad_top_k():
    resp = client.post("/match", json={
        "resume_text": "A valid resume text that is long enough to pass validation.",
        "top_k": 0,
    })
    assert resp.status_code == 422


def test_metrics_endpoint_reports_counts():
    before = client.get("/metrics").get_json()["requests_total"]
    client.post("/match", json={"resume_text": "Another valid resume text for testing purposes here."})
    after = client.get("/metrics").get_json()["requests_total"]
    assert after == before + 1
