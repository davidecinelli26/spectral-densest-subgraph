import networkx as nx
import numpy as np
import pytest

from densest import Graph, k_core
from densest.datasets import DATASETS, parse_edges
from densest.exact import core_reduction


def test_from_edges_builds_a_simple_graph():
    g = Graph.from_edges(np.array([5, 5, 7, 7, 9]), np.array([7, 7, 5, 7, 5]))  # duplicates, reverse, loop
    assert (g.n, g.m) == (3, 2)
    assert g.labels.tolist() == [5, 7, 9]
    assert g.degrees.tolist() == [2, 1, 1]
    assert g.density(np.array([0, 1])) == pytest.approx(0.5)


def test_subgraph_and_edges_within():
    g = Graph.from_edges(np.array([0, 1, 2, 2]), np.array([1, 2, 0, 3]))  # triangle + pendant
    assert g.edges_within(np.array([0, 1, 2])) == 3
    sub, idx = g.subgraph(np.array([0, 1, 2]))
    assert (sub.n, sub.m) == (3, 3) and idx.tolist() == [0, 1, 2]


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_k_core_matches_networkx(k):
    nxg = nx.gnm_random_graph(200, 700, seed=k)
    u, v = map(np.array, zip(*nxg.edges()))
    g = Graph.from_edges(u, v)
    expected = set(nx.k_core(nxg, k).nodes())
    assert {int(x) for x in g.labels[k_core(g, k)]} == expected


def test_core_reduction_keeps_the_optimum():
    nxg = nx.gnm_random_graph(150, 600, seed=3)
    u, v = map(np.array, zip(*nxg.edges()))
    g = Graph.from_edges(u, v)
    core, _, lower = core_reduction(g)
    assert core.n <= g.n
    assert core.degrees.min() >= int(np.ceil(lower.density))


def test_parse_snap_formats():
    txt = "# Directed graph\n# FromNodeId\tToNodeId\tRelationship\n1\t2\t-1\n2\t1\t1\n2\t3\t0\n3\t3\t0\n"
    g = parse_edges(txt)
    assert (g.n, g.m) == (3, 2)  # direction, relationship column and self-loop dropped
    csv = "node_1,node_2\n0,1\n1,2\n0,2\n"
    assert (parse_edges(csv, csv=True).m) == 3
    spaces = "0 1\n0 2\n"
    assert parse_edges(spaces).m == 2


def test_registry():
    assert len(DATASETS) == 15
    assert all(len(src.sha256) == 64 for src in DATASETS.values())
