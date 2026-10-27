"""Exact algorithms: Goldberg's max-flow method and Charikar's LP.

Both start from a reduction that preserves optimality. In a densest subgraph
S* every node has degree at least d(S*) inside S* (removing a node of smaller
degree would raise the density), so S* lies in the ceil(d)-core for any
lower bound d <= d(S*). Greedy peeling gives such a bound cheaply, and on real
graphs the core is a small fraction of the graph.

Goldberg (1984). For a guess g, build a network with a source s, a sink t and
    c(s, v) = U,   c(v, t) = U + 2g - deg(v),   c(u, v) = c(v, u) = 1,
with U >= max degree. A cut with source side {s} + S costs
    U n + 2 (g |S| - |E(S)|),
so a set denser than g exists iff the minimum cut is below U n, and the
minimum cut finds the set maximising |E(S)| - g |S|. Instead of the original
binary search on g, the guess is updated to the density of the set found
(Dinkelbach's method), which reaches the optimum in a few cuts. Capacities are
scaled by the denominator of g so that the max-flow runs on integers.

Charikar (2000). The LP
    max sum_e x_e   s.t.  x_e <= y_u, x_e <= y_v (e = uv),  sum_v y_v = 1,  x, y >= 0
has optimum d(S*), and the best threshold set {v : y_v >= r} attains it.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from scipy.sparse.csgraph import breadth_first_order, maximum_flow

from densest.graph import Graph, Solution, best_prefix, k_core
from densest.greedy import greedy_peeling


def core_reduction(g: Graph) -> tuple[Graph, np.ndarray, Solution]:
    """The ceil(d)-core for the greedy lower bound d, its node indices in g, and the greedy solution."""
    lower = greedy_peeling(g)
    k = -(-lower.edges // max(lower.size, 1))  # ceil(edges / size), in integers
    core, idx = g.subgraph(k_core(g, k))
    return core, idx, lower


def _densest_cut(g: Graph, edges: int, size: int) -> np.ndarray:
    """Nodes S maximising |E(S)| - g |S| for g = edges / size (empty if none beats g)."""
    n, s, t = g.n, g.n, g.n + 1
    big_u = int(g.degrees.max())
    deg = g.degrees
    rows = np.concatenate([np.full(n, s), np.arange(n), g.rows])
    cols = np.concatenate([np.arange(n), np.full(n, t), g.indices])
    cap = np.concatenate(
        [np.full(n, big_u * size), big_u * size + 2 * edges - deg * size, np.full(g.indices.size, size)]
    ).astype(np.int64)
    capacity = sp.csr_matrix((cap, (rows, cols)), shape=(n + 2, n + 2))
    flow = maximum_flow(capacity, s, t, method="dinic").flow
    residual = (capacity - flow).tocsr()
    residual.data = (residual.data > 0).astype(np.int8)
    residual.eliminate_zeros()
    source_side = breadth_first_order(residual, s, directed=True, return_predecessors=False)
    return np.sort(source_side[source_side < n])


def goldberg(g: Graph, *, prune: bool = True) -> Solution:
    """Exact densest subgraph by min cuts (Goldberg's network, Dinkelbach updates)."""
    if g.m == 0:
        return Solution(np.array([0]) if g.n else np.empty(0, np.int64), 0)
    if prune:
        core, idx, best = core_reduction(g)
    else:
        core, idx, best = g, np.arange(g.n), Solution.of(g, np.ones(g.n, dtype=bool))
    edges, size = best.edges, best.size
    while True:
        cand = _densest_cut(core, edges, size)
        if cand.size == 0:
            return best
        cand_edges = core.edges_within(cand)
        if cand_edges * size <= edges * cand.size:  # no strict improvement
            return best
        edges, size = cand_edges, cand.size
        best = Solution(np.sort(idx[cand]), cand_edges)


def charikar_lp(g: Graph, *, prune: bool = True, time_limit: float | None = None) -> tuple[Solution, float]:
    """Exact densest subgraph from Charikar's LP (HiGHS). Returns the solution and the LP optimum.

    Raises TimeoutError if HiGHS hits `time_limit` seconds."""
    if g.m == 0:
        return Solution(np.array([0]) if g.n else np.empty(0, np.int64), 0), 0.0
    core, idx = (core_reduction(g)[:2]) if prune else (g, np.arange(g.n))
    u, v = core.edge_list()
    m, n = u.size, core.n
    e = np.arange(m)
    # variables: x_0..x_{m-1}, y_0..y_{n-1}
    a_ub = sp.csr_matrix(
        (np.r_[np.ones(2 * m), -np.ones(2 * m)], (np.r_[e, e + m, e, e + m], np.r_[e, e, m + u, m + v])),
        shape=(2 * m, m + n),
    )
    a_eq = sp.csr_matrix((np.ones(n), (np.zeros(n, int), m + np.arange(n))), shape=(1, m + n))
    res = linprog(
        np.r_[-np.ones(m), np.zeros(n)], A_ub=a_ub, b_ub=np.zeros(2 * m), A_eq=a_eq, b_eq=[1.0],
        bounds=(0, None), method="highs", options={"time_limit": time_limit} if time_limit else None,
    )  # fmt: skip
    if res.status == 1:
        raise TimeoutError(f"LP not solved within {time_limit} s")
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")
    y = res.x[m:]
    sol = best_prefix(core, np.argsort(-y, kind="stable"))
    return Solution(np.sort(idx[sol.nodes]), sol.edges), float(-res.fun)
