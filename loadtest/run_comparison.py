"""
Runs the SAME Gunicorn command each Dockerfile uses (UvicornWorker for
FastAPI, sync workers for Flask) directly on the host, points a headless
Locust run at each in turn, and charts the throughput/latency comparison.

This sandbox's network policy blocks pulling base container images, so the
comparison below runs the exact Gunicorn process each container would run,
just without the container wrapper — the serving-stack comparison itself
(async event-loop workers vs. sync workers) is unaffected by that.
"""

import csv
import os
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

DURATION_S = 20
USERS = 30
SPAWN_RATE = 10

SERVICES = [
    {
        "name": "FastAPI + Gunicorn (UvicornWorker x2)",
        "port": 9001,
        "cmd": ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2",
                "-b", "127.0.0.1:9001", "--timeout", "30", "app_fastapi.main:app"],
        "csv_prefix": "fastapi_run",
    },
    {
        "name": "Flask + Gunicorn (sync workers x4)",
        "port": 9002,
        "cmd": ["gunicorn", "-w", "4", "-b", "127.0.0.1:9002", "--timeout", "30", "app_flask.app:app"],
        "csv_prefix": "flask_run",
    },
]


def wait_for_health(port, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=1)
            if r.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.3)
    return False


def run_locust(port, csv_prefix):
    cmd = [
        sys.executable, "-m", "locust",
        "-f", os.path.join(BASE, "loadtest", "locustfile.py"),
        "--host", f"http://127.0.0.1:{port}",
        "--headless",
        "-u", str(USERS),
        "-r", str(SPAWN_RATE),
        "-t", f"{DURATION_S}s",
        "--csv", os.path.join(OUT, csv_prefix),
        "--csv-full-history",
    ]
    subprocess.run(cmd, cwd=BASE, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def parse_stats(csv_prefix):
    stats_path = os.path.join(OUT, f"{csv_prefix}_stats.csv")
    with open(stats_path) as f:
        rows = list(csv.DictReader(f))
    agg = next(r for r in rows if r["Name"] == "Aggregated")
    return {
        "request_count": int(agg["Request Count"]),
        "failure_count": int(agg["Failure Count"]),
        "rps": float(agg["Requests/s"]),
        "median_ms": float(agg["Median Response Time"]),
        "p95_ms": float(agg["95%"]),
        "p99_ms": float(agg["99%"]),
        "avg_ms": float(agg["Average Response Time"]),
    }


def main():
    results = {}
    for svc in SERVICES:
        print(f"--- Starting {svc['name']} on port {svc['port']} ---")
        proc = subprocess.Popen(svc["cmd"], cwd=BASE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            if not wait_for_health(svc["port"]):
                raise RuntimeError(f"{svc['name']} did not become healthy in time")
            print(f"Healthy. Running Locust for {DURATION_S}s with {USERS} users...")
            run_locust(svc["port"], svc["csv_prefix"])
            stats = parse_stats(svc["csv_prefix"])
            results[svc["name"]] = stats
            print(f"{svc['name']}: {stats}")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            time.sleep(1)

    # --- Chart: RPS and p95 latency side by side ---
    names = list(results.keys())
    short_names = ["FastAPI\n(Uvicorn workers)", "Flask\n(sync workers)"]
    rps = [results[n]["rps"] for n in names]
    p95 = [results[n]["p95_ms"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    colors = ["#0C4DA2", "#D9A400"]

    axes[0].bar(short_names, rps, color=colors)
    axes[0].set_ylabel("Requests / second")
    axes[0].set_title("Throughput")
    for i, v in enumerate(rps):
        axes[0].text(i, v, f"{v:.1f}", ha="center", va="bottom", fontweight="bold")

    axes[1].bar(short_names, p95, color=colors)
    axes[1].set_ylabel("p95 latency (ms)")
    axes[1].set_title("Latency (95th percentile)")
    for i, v in enumerate(p95):
        axes[1].text(i, v, f"{v:.0f}ms", ha="center", va="bottom", fontweight="bold")

    fig.suptitle(f"FastAPI vs Flask — {USERS} concurrent users, {DURATION_S}s each, same model + host", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, "throughput_comparison.png"), dpi=150)

    with open(os.path.join(OUT, "throughput_comparison.md"), "w") as f:
        f.write("# Load Test Comparison\n\n")
        f.write(f"{USERS} concurrent users, {DURATION_S}s duration, identical model and host for both runs.\n\n")
        f.write("| Service | Requests | Failures | Req/s | Avg (ms) | Median (ms) | p95 (ms) | p99 (ms) |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for n in names:
            s = results[n]
            f.write(f"| {n} | {s['request_count']} | {s['failure_count']} | {s['rps']:.1f} | "
                    f"{s['avg_ms']:.1f} | {s['median_ms']:.1f} | {s['p95_ms']:.1f} | {s['p99_ms']:.1f} |\n")

    print("\nSaved outputs/throughput_comparison.png and outputs/throughput_comparison.md")


if __name__ == "__main__":
    main()
