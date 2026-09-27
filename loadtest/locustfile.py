"""
Locust load test for the /match endpoint. Run headless against each
service (see run_comparison.py) to produce an honest throughput/latency
comparison between the FastAPI+Uvicorn-worker stack and the Flask+sync-
worker stack.
"""

import random

from locust import HttpUser, task, between

SAMPLE_RESUMES = [
    "One year of experience deploying machine learning models to production using "
    "Docker, Docker Compose, FastAPI, Flask, and Gunicorn. Comfortable with Python, "
    "REST APIs, and CI/CD pipelines.",
    "Mechanical engineer with capital project experience in refining and process "
    "industries, mass and energy balances, vendor management, and EPC schedules.",
    "Site reliability engineer focused on container orchestration, observability, "
    "and tuning application servers for high-throughput production workloads.",
    "Business analyst building dashboards and cost tradeoff reports for capital "
    "project reviews, strong SQL and data visualization skills.",
    "Full stack developer shipping React frontends and Flask/FastAPI backends, "
    "deployed with Docker Compose behind an Nginx reverse proxy.",
]


class JobFitMatcherUser(HttpUser):
    wait_time = between(0.01, 0.05)

    @task
    def match(self):
        resume_text = random.choice(SAMPLE_RESUMES)
        self.client.post("/match", json={"resume_text": resume_text, "top_k": 5}, name="/match")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
