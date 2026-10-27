"""Simple undirected graphs in compressed sparse row (CSR) form.

The densest subgraph problem is defined on simple graphs: self-loops and
parallel edges are dropped when a graph is built, and the density of a node
set S is d(S) = |E(S)| / |S|, with E(S) the edges with both endpoints in S.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class Graph:
    indptr: np.ndarray  # (n + 1,) row pointers
    indices: np.ndarray  # (2m,) neighbours; every edge appears in both rows
    labels: np.ndarray = field(repr=False)  # (n,) original node identifiers

    @classmethod
    def from_edges(cls, u, v) -> Graph:
        """Build from two arrays of endpoint labels (any integers)."""
        u, v = np.asarray(u), np.asarray(v)
        labels, inv = np.unique(np.concatenate([u, v]), return_inverse=True)
        a, b = inv[: u.size], inv[u.size :]
        keep = a != b  # drop self-loops
        a, b = np.minimum(a[keep], b[keep]), np.maximum(a[keep], b[keep])
        pairs = np.unique(np.stack([a, b], axis=1), axis=0)  # drop parallel edges
        n = labels.size
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        adj = sp.csr_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(n, n))
        adj.sort_indices()
        return cls(adj.indptr.astype(np.int64), adj.indices.astype(np.int64), labels)

    @property
    def n(self) -> int:
        return self.indptr.size - 1

    @property
    def m(self) -> int:
        return self.indices.size // 2

    @cached_property
    def degrees(self) -> np.ndarray:
        return np.diff(self.indptr)

    @cached_property
    def rows(self) -> np.ndarray:
        """Row index of every entry of `indices`: (rows[i], indices[i]) is a directed edge."""
        return np.repeat(np.arange(self.n), self.degrees)

    def edge_list(self) -> tuple[np.ndarray, np.ndarray]:
        """Each undirected edge once, as (u, v) with u < v."""
        once = self.rows < self.indices
        return self.rows[once], self.indices[once]

    def adjacency(self, dtype=np.float64) -> sp.csr_matrix:
        data = np.ones(self.indices.size, dtype=dtype)
        return sp.csr_matrix((data, self.indices, self.indptr), shape=(self.n, self.n))

    def edges_within(self, nodes: np.ndarray) -> int:
        """|E(S)| for a node set given as indices or as a boolean mask."""
        mask = _as_mask(nodes, self.n)
        return int((mask[self.rows] & mask[self.indices]).sum()) // 2

    def density(self, nodes: np.ndarray) -> float:
        mask = _as_mask(nodes, self.n)
        size = int(mask.sum())
        return self.edges_within(mask) / size if size else 0.0

    def subgraph(self, nodes: np.ndarray) -> tuple[Graph, np.ndarray]:
        """Induced subgraph and, for each of its nodes, the index in this graph."""
        idx = np.flatnonzero(_as_mask(nodes, self.n))
        sub = self.adjacency(np.int8)[idx][:, idx].tocsr()
        sub.sort_indices()
        return Graph(sub.indptr.astype(np.int64), sub.indices.astype(np.int64), self.labels[idx]), idx


def _as_mask(nodes: np.ndarray, n: int) -> np.ndarray:
    nodes = np.asarray(nodes)
    if nodes.dtype == bool:
        return nodes
    mask = np.zeros(n, dtype=bool)
    mask[nodes] = True
    return mask


def k_core(g: Graph, k: int) -> np.ndarray:
    """Boolean mask of the k-core: the largest subgraph whose nodes all have degree >= k."""
    alive = np.ones(g.n, dtype=bool)
    deg = g.degrees.copy()
    while True:
        drop = alive & (deg < k)
        if not drop.any():
            return alive
        alive &= ~drop
        deg -= np.bincount(g.indices[drop[g.rows]], minlength=g.n)


@dataclass(frozen=True)
class Solution:
    """A node set of a graph, with its number of internal edges."""

    nodes: np.ndarray  # sorted node indices
    edges: int

    @property
    def size(self) -> int:
        return int(self.nodes.size)

    @property
    def density(self) -> float:
        return self.edges / self.size if self.size else 0.0

    @classmethod
    def of(cls, g: Graph, nodes: np.ndarray) -> Solution:
        idx = np.flatnonzero(_as_mask(nodes, g.n))
        return cls(idx, g.edges_within(idx))


def best_prefix(g: Graph, order: np.ndarray) -> Solution:
    """Densest prefix of a node ordering, in O(m + n).

    Adding the nodes one by one in `order`, an edge becomes internal when its
    later endpoint is added; a cumulative sum then gives |E| of every prefix.
    This is the rounding step of Charikar's LP and of the spectral method:
    sweeping a threshold over a score is the same as sweeping prefixes of the
    nodes sorted by that score.
    """
    pos = np.empty(g.n, dtype=np.int64)
    pos[order] = np.arange(g.n)
    u, v = g.edge_list()
    edges = np.cumsum(np.bincount(np.maximum(pos[u], pos[v]), minlength=g.n))
    k = int(np.argmax(edges / np.arange(1, g.n + 1)))
    return Solution(np.sort(order[: k + 1]), int(edges[k]))
