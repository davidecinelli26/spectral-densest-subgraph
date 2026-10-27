"""Peeling algorithms.

Greedy peeling (Asahiro et al. 2000; Charikar 2000) repeatedly removes a
node of minimum degree and returns the densest of the n nested sets it goes
through: a 1/2-approximation in O(m log n) with a binary heap.

Greedy++ (Boob et al. 2020) repeats the peeling T times; each node carries a
load, increased by its degree at the moment it is removed, and the node to
remove is the one of minimum load + current degree. The first pass is plain
greedy peeling, and the best density converges to the optimum as T grows.
"""

from __future__ import annotations

import heapq

import numpy as np

from densest.graph import Graph, Solution


def _peel(g: Graph, loads: list[int] | None) -> tuple[list[int], int]:
    """One peeling pass. Returns the removal order and how many removals precede the densest set.

    With `loads`, nodes are ranked by load + degree and the loads are updated
    in place (Greedy++); without, by degree alone (greedy peeling).
    """
    indptr, indices = g.indptr.tolist(), g.indices.tolist()
    deg = g.degrees.tolist()
    load = loads if loads is not None else [0] * g.n
    alive = [True] * g.n
    heap = [(load[v] + deg[v], v) for v in range(g.n)]
    heapq.heapify(heap)

    edges, size = g.m, g.n
    best_density, best_removed = edges / size, 0
    order: list[int] = []
    while heap:
        key, v = heapq.heappop(heap)
        if not alive[v] or key != load[v] + deg[v]:
            continue  # stale heap entry
        alive[v] = False
        order.append(v)
        if loads is not None:
            load[v] += deg[v]
        edges -= deg[v]
        size -= 1
        for u in indices[indptr[v] : indptr[v + 1]]:
            if alive[u]:
                deg[u] -= 1
                heapq.heappush(heap, (load[u] + deg[u], u))
        if size and edges / size > best_density:
            best_density, best_removed = edges / size, len(order)
    return order, best_removed


def _remaining(g: Graph, order: list[int], removed: int) -> Solution:
    keep = np.ones(g.n, dtype=bool)
    keep[order[:removed]] = False
    return Solution.of(g, keep)


def greedy_peeling(g: Graph) -> Solution:
    """1/2-approximation of the densest subgraph."""
    if g.n == 0:
        return Solution(np.empty(0, np.int64), 0)
    order, removed = _peel(g, loads=None)
    return _remaining(g, order, removed)


def greedy_plus_plus(g: Graph, iterations: int) -> Solution:
    """Greedy++ with `iterations` peeling passes; returns the densest set seen."""
    if g.n == 0:
        return Solution(np.empty(0, np.int64), 0)
    loads = [0] * g.n
    best = None
    for _ in range(max(iterations, 1)):
        order, removed = _peel(g, loads)
        sol = _remaining(g, order, removed)
        if best is None or sol.density > best.density:
            best = sol
    return best
