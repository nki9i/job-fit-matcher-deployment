"""
Shared inference module — loaded once at process startup by both the
FastAPI and Flask services. Keeping this identical between the two apps is
the point: the two services are a controlled comparison of *serving stack*
(async/Gunicorn+Uvicorn vs. sync/Gunicorn), not a comparison of two
different models.
"""

import json
import os
import time

import joblib
import numpy as np

ARTIFACTS_DIR = os.environ.get(
    "ARTIFACTS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "artifacts"),
)

MODEL_VERSION = "job-fit-matcher-v1"


class JobFitMatcher:
    def __init__(self, artifacts_dir: str = ARTIFACTS_DIR):
        self.artifacts_dir = artifacts_dir
        self.vectorizer = None
        self.svd = None
        self.posting_embeddings = None
        self.postings_metadata = None
        self.posting_top_terms = None
        self.loaded_at = None

    def load(self):
        t0 = time.time()
        self.vectorizer = joblib.load(os.path.join(self.artifacts_dir, "vectorizer.joblib"))
        self.svd = joblib.load(os.path.join(self.artifacts_dir, "svd.joblib"))
        self.posting_embeddings = np.load(os.path.join(self.artifacts_dir, "posting_embeddings.npy"))
        with open(os.path.join(self.artifacts_dir, "postings_metadata.json")) as f:
            self.postings_metadata = json.load(f)
        with open(os.path.join(self.artifacts_dir, "posting_top_terms.json")) as f:
            self.posting_top_terms = json.load(f)
        self.loaded_at = time.time()
        return time.time() - t0

    @property
    def is_ready(self) -> bool:
        return self.posting_embeddings is not None

    def _embed_query(self, text: str) -> np.ndarray:
        tfidf_vec = self.vectorizer.transform([text])
        emb = self.svd.transform(tfidf_vec)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb.ravel()

    def _query_top_terms(self, text: str, k: int = 10) -> list[str]:
        tfidf_vec = self.vectorizer.transform([text]).toarray().ravel()
        feature_names = np.asarray(self.vectorizer.get_feature_names_out())
        top_idx = tfidf_vec.argsort()[::-1][:k]
        return [feature_names[i] for i in top_idx if tfidf_vec[i] > 0]

    def rank_postings(self, resume_text: str, top_k: int = 5) -> list[dict]:
        if not self.is_ready:
            raise RuntimeError("Model not loaded — call .load() first")

        query_emb = self._embed_query(resume_text)
        scores = self.posting_embeddings @ query_emb  # cosine similarity (vectors are L2-normalized)

        query_terms = set(self._query_top_terms(resume_text))
        order = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(order, start=1):
            meta = self.postings_metadata[idx]
            posting_terms = set(self.posting_top_terms[idx])
            matched_keywords = sorted(query_terms & posting_terms)[:6]
            results.append({
                "rank": rank,
                "posting_id": meta["id"],
                "title": meta["title"],
                "company": meta["company"],
                "location": meta["location"],
                "score": round(float(scores[idx]), 4),
                "matched_keywords": matched_keywords,
            })
        return results


# Module-level singleton — both apps import this same instance so the
# model is loaded exactly once per worker process.
matcher = JobFitMatcher()
