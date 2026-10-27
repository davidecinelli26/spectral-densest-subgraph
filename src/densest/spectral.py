"""Spectral-Charikar: Charikar's rounding applied to the leading eigenvector.

Charikar's LP gives every node a weight y_v, and some threshold set
{v : y_v >= r} is optimal. Spectral-Charikar replaces the LP solution with the
eigenvector of the largest eigenvalue of the adjacency matrix, which costs one
sparse eigensolve instead of an LP with m + n variables, and keeps the best
threshold set.

For a connected graph that eigenvector is the Perron vector, with entries of
one sign; on a disconnected graph it lives on a single component and is
numerically zero elsewhere. Its absolute value is therefore the score.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import eigsh

from densest.graph import Graph, Solution, best_prefix


def leading_eigenvector(g: Graph, seed: int = 0) -> np.ndarray:
    """|u_max|: absolute value of the eigenvector of the largest adjacency eigenvalue."""
    if g.n <= 2:
        return np.ones(g.n)
    v0 = np.random.default_rng(seed).uniform(0.5, 1.5, g.n)  # fixed start: deterministic result
    _, vec = eigsh(g.adjacency(), k=1, which="LA", v0=v0)
    return np.abs(vec[:, 0])


def spectral_charikar(g: Graph, *, decimals: int | None = None, seed: int = 0) -> Solution:
    """Densest threshold set of the leading eigenvector.

    By default every distinct threshold is tried (a sweep over all prefixes,
    O(m + n log n)). With `decimals`, only thresholds rounded down to that many
    decimals are tried, as in the thesis code: on large graphs the entries are
    of order 1/sqrt(n), so three decimals leave only a handful of thresholds.
    """
    if g.n == 0:
        return Solution(np.empty(0, np.int64), 0)
    score = leading_eigenvector(g, seed)
    order = np.argsort(-score, kind="stable")
    if decimals is None:
        return best_prefix(g, order)

    # threshold r keeps {v : score_v >= r}, a prefix of `order`
    best = Solution(np.empty(0, np.int64), 0)
    for r in np.unique(np.floor(score * 10**decimals) / 10**decimals):
        sol = Solution.of(g, score >= r)
        if sol.density > best.density:
            best = sol
    return best


def refine(g: Graph, sol: Solution, max_moves: int | None = None) -> Solution:
    """Local search from `sol`: move one node at a time while the density strictly increases.

    Adding v raises the density iff deg_S(v) > d(S); removing v iff deg_S(v) < d(S).
    Each step makes the move with the largest margin, so the density increases
    monotonically and the result is never worse than the input. The degrees
    towards S are updated incrementally: a move costs O(n + deg(v)).
    """
    mask = np.zeros(g.n, dtype=bool)
    mask[sol.nodes] = True
    inside = g.adjacency(np.int64) @ mask.astype(np.int64)  # degree towards S of every node
    edges, size = sol.edges, sol.size
    for _ in range(max_moves or g.n):
        gain = np.where(mask, edges - inside * size, inside * size - edges)  # > 0 iff the move helps
        if size <= 1:
            gain[mask] = -1
        v = int(np.argmax(gain))
        if gain[v] <= 0:
            break
        nbrs = g.indices[g.indptr[v] : g.indptr[v + 1]]
        step = -1 if mask[v] else 1
        edges += step * int(inside[v])
        size += step
        mask[v] = step > 0
        inside[nbrs] += step
    return Solution(np.flatnonzero(mask), edges)


def spectral_refined(g: Graph, seed: int = 0) -> Solution:
    """Spectral-Charikar followed by the local search."""
    return refine(g, spectral_charikar(g, seed=seed))
