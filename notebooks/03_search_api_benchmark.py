# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB3 — FastAPI `/search` Endpoint + Latency Benchmark
#
# **Stack:** FastAPI + uvicorn + httpx (client). Searcher từ `app/search.py`.
# Maps to slide §7 (Production Patterns) + deliverable bullets 1, 4.
#
# > Mục tiêu: bọc `Searcher` thành REST API, đo P50/P95/P99 latency, đảm bảo
# > P99 < 50 ms cho hybrid mode (rubric threshold).

# %%
import _setup  # noqa: F401
import atexit
import os
import statistics
import subprocess
import time
from pathlib import Path

import httpx

# %% [markdown]
# ## 1. Khởi động API server (background)
#
# Trong production thực tế, bạn sẽ chạy `make api` ở terminal riêng. Notebook
# này khởi động uvicorn ở background subprocess và đợi `/healthz` trả ready.

# %%
ROOT = Path(_setup.__file__).resolve().parent.parent
server_env = os.environ.copy()
server_env["SEARCH_QUERY_CACHE"] = "0"
server_env["EMBEDDING_THREADS"] = "1"
# Keep the single-worker benchmark deterministic on constrained CI runners;
# otherwise BLAS/OpenMP can oversubscribe the embedding request path.
server_env["OMP_NUM_THREADS"] = "1"
server_env["OPENBLAS_NUM_THREADS"] = "1"
server_env["MKL_NUM_THREADS"] = "1"
server_env["PYTHONPATH"] = str(ROOT) + os.pathsep + server_env.get("PYTHONPATH", "")
proc = subprocess.Popen(
    ["uvicorn", "app.main:app", "--port", "8000", "--log-level", "warning"],
    cwd=str(ROOT),
    env=server_env,
)


def stop_server() -> None:
    if proc.poll() is None:
        proc.terminate()


atexit.register(stop_server)

# Đợi server up + warm (Searcher.from_corpus loads embeddings + indexes 1000 docs).
# The primary measurement disables only the query-vector cache: model, index,
# and HTTP server are still warm, while golden queries pay realistic embedding
# cost instead of becoming cache hits during the benchmark.
URL = "http://localhost:8000"
for _ in range(600):
    try:
        r = httpx.get(f"{URL}/healthz", timeout=2.0)
        if r.status_code == 200 and r.json().get("ready"):
            break
    except httpx.HTTPError:
        pass
    time.sleep(1)
else:
    raise RuntimeError("API didn't become ready within 600s")

print(httpx.get(f"{URL}/healthz").json())

# %% [markdown]
# ## 2. Single query — kiểm tra response shape

# %%
r = httpx.get(f"{URL}/search", params={"q": "cloud computing tự động mở rộng", "mode": "hybrid"})
r.raise_for_status()
body = r.json()
print(f"latency_ms: {body['latency_ms']:.1f}")
print(f"top-3 hits:")
for h in body["hits"][:3]:
    print(f"  {h['doc_id']:>14}  score={h['score']:.4f}  {h['title']}")

# %% [markdown]
# ## 3. Latency benchmark (500 queries × 3 modes)
#
# Dùng 50 golden queries × 10 reps = 500 calls/mode. Ghi nhận latency từ
# `body["latency_ms"]` (server-side, đã trừ network) HOẶC từ wall-clock httpx
# (bao gồm network) — note: rubric assert P99 < 50ms áp dụng cho server-side.
#
# Output: bảng P50/P95/P99 cho 3 mode.

# %%
import json

DATA = ROOT / "data"
golden = [json.loads(l) for l in (DATA / "golden_set.jsonl").open(encoding="utf-8")]


def percentile(values: list[float], p: float) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    return sorted(values)[min(int(n * p), n - 1)]


def benchmark_mode(mode: str, reps: int = 10) -> dict[str, float]:
    server_latencies: list[float] = []
    wall_latencies: list[float] = []
    for _ in range(reps):
        for q in golden:
            t0 = time.perf_counter()
            r = httpx.get(f"{URL}/search", params={"q": q["query"], "mode": mode})
            r.raise_for_status()
            wall_latencies.append((time.perf_counter() - t0) * 1000)
            server_latencies.append(r.json()["latency_ms"])
    return {
        "p50_server": percentile(server_latencies, 0.50),
        "p95_server": percentile(server_latencies, 0.95),
        "p99_server": percentile(server_latencies, 0.99),
        "p99_wall": percentile(wall_latencies, 0.99),
    }


# Warm infrastructure with probes that are not in the benchmark set. This
# removes model/index/server startup from the measurement without pre-populating
# the cache with the exact golden queries.
warmup_queries = [
    "kiểm tra khởi động máy chủ tìm kiếm",
    "đo độ sẵn sàng của chỉ mục vector",
    "warmup truy vấn HTTP không thuộc golden set",
]
for mode in ("keyword", "semantic", "hybrid"):
    for warm_query in warmup_queries:
        warm = httpx.get(f"{URL}/search", params={"q": warm_query, "mode": mode})
        warm.raise_for_status()
print(
    f"Infrastructure warm-up complete: {len(warmup_queries) * 3} successful HTTP requests; "
    "query-vector cache disabled for primary measurements"
)

print(f"  {'mode':10}  {'P50':>7}  {'P95':>7}  {'P99':>7}  {'P99(wall)':>9}")
results = {}
for mode in ("keyword", "semantic", "hybrid"):
    res = benchmark_mode(mode)
    results[mode] = res
    print(f"  {mode:10}  {res['p50_server']:>5.1f}ms  {res['p95_server']:>5.1f}ms  "
          f"{res['p99_server']:>5.1f}ms  {res['p99_wall']:>7.1f}ms")

# %% [markdown]
# ## 4. Rubric assertion — hybrid P99 server-side < 50ms

# %%
hybrid_p99 = results["hybrid"]["p99_server"]
print(f"Hybrid P99 server-side: {hybrid_p99:.1f}ms")
assert hybrid_p99 < 50, (
    f"Hybrid P99 server-side must be < 50ms after warm-up; got {hybrid_p99:.1f}ms"
)
print(f"PASS — hybrid P99 < 50ms ({hybrid_p99:.1f}ms)")

# %% [markdown]
# ## 5. Cleanup — stop the API server

# %%
stop_server()
proc.wait(timeout=5)
print("API server stopped")

# %% [markdown]
# ## Deliverable evidence
#
# 1. Output cell 2: 1 single hybrid query response with `top-3 hits`.
# 2. Output cell 3: latency table P50/P95/P99 for keyword/semantic/hybrid.
# 3. Output cell 4: hybrid P99 < 50ms PASS.
#
# ---
#
# ## Vibe-coding callout
#
# **Delegate freely:** the FastAPI scaffolding (route definition, Pydantic
# response model, lifespan handler). AI generates this perfectly given the
# spec "GET /search?q=str&mode=Literal[...] returning SearchResponse with
# latency_ms field". `app/main.py` is exactly that pattern — review the diff,
# don't write it from scratch.
#
# **Think hard yourself:** *what to measure*. Server-side latency vs wall-clock
# vs client-side. P50 vs P95 vs P99. Cold vs warm. Single user vs concurrent.
# These are *judgement* decisions: nếu rubric chỉ check P99, optimization sẽ
# hướng vào tail latency, không phải mean. Đừng nhờ AI quyết định metric —
# chỉ nhờ implement metric đã chọn.
