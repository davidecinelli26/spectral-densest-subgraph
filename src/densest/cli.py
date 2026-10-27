"""Command line: `densest download | solve | benchmark | report`."""

from __future__ import annotations

import argparse
import time

from densest import datasets
from densest.benchmark import ALGORITHMS


def _download(args) -> None:
    for name in args.datasets:
        datasets.download(name, args.raw_dir)
    print(f"{len(args.datasets)} datasets ready in {args.raw_dir}")


def _solve(args) -> None:
    g = datasets.load(args.dataset, args.raw_dir)
    print(f"{args.dataset}: {g.n:,} nodes, {g.m:,} edges, density {g.m / g.n:.3f}")
    for algo in args.algorithms:
        start = time.perf_counter()
        sol = ALGORITHMS[algo](g)
        print(f"  {algo:16s} density {sol.density:10.4f}  |S| {sol.size:7,d}  {time.perf_counter() - start:8.3f}s")


def _benchmark(args) -> None:
    from densest.benchmark import run

    run(args.datasets, args.algorithms, raw_dir=args.raw_dir, out=f"{args.out_dir}/benchmark.csv")


def _report(args) -> None:
    from densest.report import write_report

    print(f"written {write_report(args.out_dir)}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="densest", description=__doc__)
    p.add_argument("--raw-dir", default="data/raw", help="where the SNAP archives are stored")
    p.add_argument("--out-dir", default="results", help="where benchmark.csv, summary.md and figures go")
    sub = p.add_subparsers(dest="cmd", required=True)
    names, algos = list(datasets.DATASETS), list(ALGORITHMS)

    d = sub.add_parser("download", help="download and verify the SNAP archives")
    d.add_argument("datasets", nargs="*", default=names, choices=names, metavar="DATASET")
    d.set_defaults(fn=_download)

    s = sub.add_parser("solve", help="run algorithms on one dataset")
    s.add_argument("dataset", choices=names)
    s.add_argument("-a", "--algorithms", nargs="+", default=["goldberg", "greedy", "spectral+refine"], choices=algos)
    s.set_defaults(fn=_solve)

    b = sub.add_parser("benchmark", help="all algorithms on all datasets (resumable)")
    b.add_argument("--datasets", nargs="+", default=None, choices=names)
    b.add_argument("--algorithms", nargs="+", default=None, choices=algos)
    b.set_defaults(fn=_benchmark)

    sub.add_parser("report", help="tables and figures from results/benchmark.csv").set_defaults(fn=_report)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
