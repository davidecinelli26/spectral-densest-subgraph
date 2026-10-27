# Changes since the thesis

This repository revisits the code of my bachelor's thesis, *Un algoritmo spettrale per l'estrazione di sottografi densi* (University of Calabria, A.Y. 2023/2024). The proposed method is the same, Spectral-Charikar; the comparison it is measured against was rebuilt from scratch, and the local refinement the thesis listed as future work is now implemented. The numbers in the thesis are superseded by [`results/`](results/).

## Algorithms

- **Greedy peeling.** The search for the minimum-degree node scanned the degree buckets only up to the number of distinct degrees seen so far, not up to the maximum degree, so peeling could stop early. On `as20000102` the thesis reports a 45% gap to the optimum; correctly implemented, greedy peeling is 0.16% from it. It now uses a binary heap, O(m log n).
- **Greedy++.** The scan for the node of minimum load + degree started from the degree of the last removed node, so it could miss nodes of lower degree, and on `as20000102` it returned less than half the optimal density, which a correct implementation cannot do (its first pass is greedy peeling, a 1/2-approximation). It now follows Boob et al. (2020) with a heap.
- **Spectral-Charikar.** Thresholds were the eigenvector entries rounded down to three decimals. On a graph with n nodes the entries are of order 1/√n, so on large graphs only a handful of thresholds survived, and each one was evaluated with a full O(m) pass. The method now sweeps every threshold in O(m + n log n), the cost the thesis states for it. The eigenvector comes from the symmetric solver (`eigsh`) with a fixed start vector, and its absolute value replaces the majority-sign flip. The rounded variant is kept (`spectral_3dec`) to measure the difference.
- **Goldberg.** The binary search on the density ran `networkx` max-flows on the whole graph. It now restricts the graph to the ⌈d⌉-core for the greedy lower bound d (which provably contains every densest subgraph), updates the guess to the density of the set found by each cut (Dinkelbach's method, a few cuts instead of ~40), and runs SciPy's Dinic max-flow on integer capacities.
- **Charikar's LP.** Solved with SciPy's HiGHS instead of PuLP/CBC, on the same core, with a time limit.

## New

- **Local refinement** of the spectral solution, the "refinement technique" proposed in the thesis' conclusions: one node at a time is added (if its degree towards S exceeds d(S)) or removed (if below), each move strictly increasing the density, with degrees updated incrementally.

## Data

- **Simple graphs.** The thesis read the edge lists with `networkx`, which keeps self-loops and counts them as edges: 1,323 of the 13,895 "edges" of `as20000102`, 642 in `email-Eu-core`. The densest subgraph problem is defined on simple graphs, so self-loops, parallel edges and directions are now dropped. Optimal densities therefore differ from the thesis on the graphs that have self-loops.
- **The original SNAP files**, downloaded and checked against SHA-256 checksums. The Twitch graph used in the thesis had been opened and saved with Excel, which truncated it at its 1,048,575-row limit: 811 thousand of the 6.8 million edges. The four MUSAE graphs (Deezer, LastFM, chameleon, squirrel) also differed from the SNAP release by a few thousand edges.

## Engineering

- A Python package (`src/densest`) with one CSR graph type: no `networkx` at run time.
- A `densest` command line for download, solving one graph, the full benchmark and the report.
- A resumable benchmark that writes every (dataset, algorithm) pair to `results/benchmark.csv`, with timings that take the fastest of up to five runs for sub-second algorithms.
- Tests on every push (GitHub Actions): every exact method against brute force on random graphs, the 1/2 guarantee of greedy peeling, monotonicity of the refinement, k-cores against `networkx`, and the SNAP file formats.
