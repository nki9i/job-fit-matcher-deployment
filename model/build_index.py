"""
Build Index — Job Fit Matcher
================================

Offline "training" step: fits a TF-IDF + Truncated SVD (LSA) embedding
pipeline on the job-postings corpus and serializes everything the two
production services need at inference time.

Design choice, worth defending in an interview: this uses a classical
TF-IDF + SVD pipeline rather than a downloaded transformer embedding model.
That's a deliberate production tradeoff, not a limitation of the approach:
- No external model download at build or run time -> fully reproducible,
  offline-buildable container images (no dependency on a model hub being
  reachable at deploy time).
- CPU-only, sub-second inference, no GPU required, tiny image size.
- Fully interpretable: because the embedding space is built directly from
  TF-IDF term weights, we can recover which *words* actually drove a match
  (see `matched_keywords` in search_engine.py) — something a dense
  transformer embedding does not give you for free.

Run: python model/build_index.py
Outputs (checked into the repo so containers can load them without
re-running this step):
  model/artifacts/vectorizer.joblib
  model/artifacts/svd.joblib
  model/artifacts/posting_embeddings.npy
  model/artifacts/posting_top_terms.json
  model/artifacts/postings_metadata.json
"""

import json
import os

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

BASE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(BASE, "artifacts")
os.makedirs(ARTIFACTS, exist_ok=True)

N_COMPONENTS = 40      # LSA dimensionality; comfortably < n_postings and < vocab size
TOP_TERMS_PER_DOC = 10  # for the keyword-overlap explanation feature


def load_corpus():
    with open(os.path.join(BASE, "postings_corpus.json")) as f:
        return json.load(f)


def top_terms_for_row(row, feature_names, k=TOP_TERMS_PER_DOC):
    """Return the top-k highest-weighted TF-IDF terms for one document row."""
    row = np.asarray(row).ravel()
    top_idx = row.argsort()[::-1][:k]
    return [feature_names[i] for i in top_idx if row[i] > 0]


def main():
    postings = load_corpus()
    texts = [p["text"] for p in postings]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.9,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = np.array(vectorizer.get_feature_names_out())

    n_components = min(N_COMPONENTS, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    lsa_embeddings = svd.fit_transform(tfidf_matrix)
    lsa_embeddings = normalize(lsa_embeddings)  # L2-normalize -> cosine similarity == dot product

    top_terms = [top_terms_for_row(tfidf_matrix[i].toarray(), feature_names) for i in range(len(postings))]

    joblib.dump(vectorizer, os.path.join(ARTIFACTS, "vectorizer.joblib"))
    joblib.dump(svd, os.path.join(ARTIFACTS, "svd.joblib"))
    np.save(os.path.join(ARTIFACTS, "posting_embeddings.npy"), lsa_embeddings)

    with open(os.path.join(ARTIFACTS, "posting_top_terms.json"), "w") as f:
        json.dump(top_terms, f, indent=2)

    metadata = [{"id": p["id"], "title": p["title"], "company": p["company"], "location": p["location"]}
                for p in postings]
    with open(os.path.join(ARTIFACTS, "postings_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Indexed {len(postings)} postings")
    print(f"Vocabulary size: {len(feature_names)}")
    print(f"Embedding dimensionality: {n_components}")
    print(f"Explained variance (SVD): {svd.explained_variance_ratio_.sum():.3f}")
    print(f"Artifacts written to {ARTIFACTS}/")


if __name__ == "__main__":
    main()
