"""Tables and figures from `results/benchmark.csv`."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from densest.benchmark import ALGORITHMS, EXACT

APPROX = ["greedy", "greedy++", "spectral_3dec", "spectral", "spectral+refine"]
LABELS = {
    "goldberg": "Goldberg (exact)",
    "charikar_lp": "Charikar LP (exact)",
    "greedy": "Greedy peeling",
    "greedy++": "Greedy++",
    "spectral_3dec": "Spectral-Charikar, thesis thresholds",
    "spectral": "Spectral-Charikar",
    "spectral+refine": "Spectral-Charikar + refinement",
}


def _md(df: pd.DataFrame) -> str:
    return df.to_markdown(disable_numparse=True)


def _pct(x: float) -> str:
    """Accuracy truncated, not rounded, so that 100.00% means the optimum."""
    return f"{np.floor(x * 1e4 + 1e-9) / 100:.2f}%"


def with_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    opt = df[df["algorithm"] == EXACT].set_index("dataset")["density"]
    df = df.copy()
    df["accuracy"] = df["density"] / df["dataset"].map(opt)
    return df


def datasets_table(df: pd.DataFrame) -> pd.DataFrame:
    exact = df[df["algorithm"] == EXACT].set_index("dataset")
    t = pd.DataFrame({
        "nodes": exact["n"].map("{:,}".format),
        "edges": exact["m"].map("{:,}".format),
        "density": (exact["m"] / exact["n"]).map("{:.2f}".format),
        "optimal density": exact["density"].map("{:.3f}".format),
        "densest subgraph size": exact["size"].map("{:,.0f}".format),
    })  # fmt: skip
    return t.sort_index(key=lambda s: s.str.lower())


def pivot(df: pd.DataFrame, value: str, fmt: Callable[[float], str], algorithms: list[str]) -> pd.DataFrame:
    t = df.pivot_table(index="dataset", columns="algorithm", values=value, aggfunc="first")
    t = t[[a for a in algorithms if a in t.columns]].sort_index(key=lambda s: s.str.lower())
    t = t.map(lambda x: "timeout" if pd.isna(x) else fmt(x))  # only timeouts leave a cell empty
    t.columns = [LABELS[c] for c in t.columns]
    return t


def overview(df: pd.DataFrame) -> pd.DataFrame:
    """Per algorithm: worst and mean accuracy, optima found, wins against greedy, time.

    Accuracies are over the graphs solved; a timeout counts as the time limit,
    so the times of an algorithm that timed out are lower bounds (marked ≥).
    """
    acc = df.pivot_table(index="dataset", columns="algorithm", values="accuracy", aggfunc="first")
    sec = df.pivot_table(index="dataset", columns="algorithm", values="time_s", aggfunc="first")
    timeouts = df[df["status"] == "timeout"].groupby("algorithm").size()
    rows = []
    for a in [x for x in ALGORITHMS if x in acc.columns]:
        a_acc, g_acc = acc[a].round(6), acc["greedy"].round(6)
        bound = "≥ " if timeouts.get(a, 0) else ""
        rows.append({
            "algorithm": LABELS[a],
            "worst accuracy": _pct(a_acc.min()),
            "mean accuracy": _pct(a_acc.mean()),
            "optimum found": f"{(a_acc >= 1 - 1e-9).sum()}/{len(a_acc)}",
            "vs greedy (better/equal/worse)": f"{(a_acc > g_acc).sum()}/{(a_acc == g_acc).sum()}/{(a_acc < g_acc).sum()}",
            "total time (s)": f"{bound}{sec[a].sum():.1f}",
            "geometric mean time (s)": f"{bound}{np.exp(np.log(sec[a]).mean()):.3f}",
        })  # fmt: skip
    return pd.DataFrame(rows).set_index("algorithm")


def figures(df: pd.DataFrame, out: Path) -> None:
    order = sorted(df["dataset"].unique(), key=str.lower)
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    shown = ["greedy", "spectral", "spectral+refine"]
    w = 0.8 / len(shown)
    acc = df.pivot_table(index="dataset", columns="algorithm", values="accuracy").loc[order]
    for i, a in enumerate(shown):
        ax.bar(x + (i - (len(shown) - 1) / 2) * w, 100 * (1 - acc[a]), w, label=LABELS[a])
    ax.set_xticks(x, order, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("gap to the optimal density (%)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "gap_to_optimum.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    sec = df.pivot_table(index="dataset", columns="algorithm", values="time_s").loc[order]
    for a, style in [("goldberg", "s-"), ("greedy", "o-"), ("greedy++", "^-"), ("spectral", "D-"),
                     ("spectral+refine", "v-")]:  # fmt: skip
        ax.plot(x, sec[a], style, label=LABELS[a], ms=4)
    ax.set_yscale("log")
    ax.set_xticks(x, order, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("time (s, log scale)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "runtime.png", dpi=150)
    plt.close(fig)


def write_report(results_dir: str | Path = "results") -> Path:
    results_dir = Path(results_dir)
    df = with_accuracy(pd.read_csv(results_dir / "benchmark.csv"))
    (results_dir / "figures").mkdir(exist_ok=True)
    figures(df, results_dir / "figures")
    timeouts = df[df["status"] == "timeout"]
    solved = df.assign(time_s=df["time_s"].where(df["status"] == "ok"))
    parts = [
        "# Benchmark\n",
        (
            "15 SNAP graphs as simple undirected graphs (no self-loops, no parallel edges). Accuracy is the density "
            "found divided by the optimal density computed by Goldberg's method, truncated to two decimals: "
            "100.00% means that the optimum was found. Generated by `densest report` from `benchmark.csv`.\n"
        ),
        "## Overview\n", _md(overview(df)), "",
    ]  # fmt: skip
    for algo, rows in timeouts.groupby("algorithm"):
        parts.append(
            f"{LABELS[algo]} did not finish within {rows['time_s'].iloc[0]:.0f} s on {', '.join(rows['dataset'])}: "
            "its accuracy is over the other graphs, and its times count the limit.\n"
        )
    parts += [
        "![Gap to the optimum](figures/gap_to_optimum.png)\n",
        "![Running time](figures/runtime.png)\n",
        "## Datasets\n", _md(datasets_table(df)), "",
        "## Accuracy (density / optimal density)\n", _md(pivot(df, "accuracy", _pct, APPROX)), "",
        "## Size of the subgraph found\n", _md(pivot(df, "size", "{:,.0f}".format, list(ALGORITHMS))), "",
        "## Time (s)\n", _md(pivot(solved, "time_s", "{:.3f}".format, list(ALGORITHMS))), "",
    ]  # fmt: skip
    out = results_dir / "summary.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out
