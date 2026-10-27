"""Benchmark: every algorithm on every dataset, against the exact optimum.

Times are wall-clock seconds of the whole algorithm, including its own
preprocessing (the exact methods include the greedy lower bound and the core
reduction). Algorithms that finish within a second are run up to five times
and the fastest run is kept, to limit timer noise on small graphs.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from densest import datasets
from densest.exact import charikar_lp, goldberg
from densest.graph import Graph, Solution
from densest.greedy import greedy_peeling, greedy_plus_plus
from densest.spectral import spectral_charikar, spectral_refined

LP_TIME_LIMIT = 600.0

ALGORITHMS: dict[str, Callable[[Graph], Solution]] = {
    "goldberg": goldberg,
    "charikar_lp": lambda g: charikar_lp(g, time_limit=LP_TIME_LIMIT)[0],
    "greedy": greedy_peeling,
    "greedy++": lambda g: greedy_plus_plus(g, max(1, math.floor(math.log(g.n)))),  # T = floor(ln n), as in the thesis
    "spectral_3dec": lambda g: spectral_charikar(g, decimals=3),  # thresholds of the thesis code
    "spectral": spectral_charikar,
    "spectral+refine": spectral_refined,
}
EXACT = "goldberg"
COLUMNS = ["dataset", "n", "m", "algorithm", "density", "size", "edges", "time_s", "status"]


def timed(fn: Callable[[Graph], Solution], g: Graph, budget: float = 1.0, repeats: int = 5) -> tuple[Solution, float]:
    start = time.perf_counter()
    sol = fn(g)
    best = time.perf_counter() - start
    if best < budget:
        for _ in range(repeats - 1):
            start = time.perf_counter()
            fn(g)
            best = min(best, time.perf_counter() - start)
    return sol, best


def run(
    names: list[str] | None = None,
    algorithms: list[str] | None = None,
    *,
    raw_dir: str | Path = "data/raw",
    out: str | Path = "results/benchmark.csv",
) -> pd.DataFrame:
    """Run the missing (dataset, algorithm) pairs and return the full table."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=COLUMNS)
    have = set(zip(done["dataset"], done["algorithm"]))
    for name in names or list(datasets.DATASETS):
        todo = [a for a in algorithms or list(ALGORITHMS) if (name, a) not in have]
        if not todo:
            continue
        g = datasets.load(name, raw_dir)
        for algo in todo:
            row = {"dataset": name, "n": g.n, "m": g.m, "algorithm": algo}
            try:
                sol, seconds = timed(ALGORITHMS[algo], g)
                row.update(density=sol.density, size=sol.size, edges=sol.edges, time_s=seconds, status="ok")
            except TimeoutError:
                row.update(time_s=LP_TIME_LIMIT, status="timeout")
            pd.DataFrame([row], columns=COLUMNS).to_csv(out, mode="a", header=not out.exists(), index=False)
            shown = f"density {row['density']:.4f}" if row["status"] == "ok" else "timeout"
            print(f"{name:18s} {algo:16s} {shown:20s} {row['time_s']:9.3f}s", flush=True)
    return pd.read_csv(out)
