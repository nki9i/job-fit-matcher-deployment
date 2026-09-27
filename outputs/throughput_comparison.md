# Load Test Comparison

30 concurrent users, 20s duration, identical model and host for both runs.

| Service | Requests | Failures | Req/s | Avg (ms) | Median (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|---|---|---|---|
| FastAPI + Gunicorn (UvicornWorker x2) | 13683 | 0 | 719.5 | 8.9 | 7.0 | 23.0 | 31.0 |
| Flask + Gunicorn (sync workers x4) | 14448 | 0 | 759.7 | 6.6 | 6.0 | 13.0 | 17.0 |
