import numpy as np
import pandas as pd

from densest import Graph, benchmark


def test_rows_keep_their_columns_and_the_run_resumes(tmp_path, monkeypatch):
    g = Graph.from_edges(np.array([0, 1, 0]), np.array([1, 2, 2]))
    monkeypatch.setattr(benchmark.datasets, "load", lambda name, raw_dir: g)

    def too_slow(_):
        raise TimeoutError

    monkeypatch.setitem(benchmark.ALGORITHMS, "too_slow", too_slow)
    out = tmp_path / "benchmark.csv"
    rows = benchmark.run(["ca-GrQc"], ["greedy", "too_slow"], out=out).set_index("algorithm")
    assert rows.loc["greedy", "density"] == 1.0 and rows.loc["greedy", "status"] == "ok"
    assert rows.loc["too_slow", "status"] == "timeout" and pd.isna(rows.loc["too_slow", "density"])
    assert len(benchmark.run(["ca-GrQc"], ["greedy", "too_slow"], out=out)) == 2  # nothing is run twice
