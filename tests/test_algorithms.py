"""Every algorithm against brute force on small random graphs."""

from itertools import combinations

import numpy as np
import pytest

from densest import (
    Graph,
    Solution,
    best_prefix,
    charikar_lp,
    goldberg,
    greedy_peeling,
    greedy_plus_plus,
    refine,
    spectral_charikar,
    spectral_refined,
)


def random_graph(seed: int, n: int = 10, p: float = 0.35) -> Graph:
    rng = np.random.default_rng(seed)
    u, v = np.triu_indices(n, 1)
    keep = rng.uniform(size=u.size) < p
    if not keep.any():
        keep[0] = True
    return Graph.from_edges(u[keep], v[keep])


def brute_force(g: Graph) -> float:
    return max(g.density(np.array(s)) for k in range(1, g.n + 1) for s in combinations(range(g.n), k))


GRAPHS = [random_graph(s, n=8 + s % 4, p=0.25 + 0.05 * (s % 5)) for s in range(25)]


@pytest.mark.parametrize("g", GRAPHS)
def test_exact_methods_match_brute_force(g):
    opt = brute_force(g)
    for sol in (goldberg(g), goldberg(g, prune=False), charikar_lp(g)[0], charikar_lp(g, prune=False)[0]):
        assert sol.density == pytest.approx(opt)
        assert sol.edges == g.edges_within(sol.nodes)
    assert charikar_lp(g)[1] == pytest.approx(opt)  # LP value = optimal density (Charikar 2000)


@pytest.mark.parametrize("g", GRAPHS)
def test_approximations_respect_their_guarantees(g):
    opt = brute_force(g)
    greedy = greedy_peeling(g)
    assert opt / 2 - 1e-12 <= greedy.density <= opt + 1e-12  # 1/2-approximation
    assert greedy_plus_plus(g, 1).density == pytest.approx(greedy.density)  # first pass = greedy peeling
    assert greedy.density - 1e-12 <= greedy_plus_plus(g, 8).density <= opt + 1e-12
    spectral = spectral_charikar(g)
    assert 0 < spectral.density <= opt + 1e-12
    refined = refine(g, spectral)
    assert spectral.density - 1e-12 <= refined.density <= opt + 1e-12
    assert spectral_refined(g).density == pytest.approx(refined.density)
    for sol in (greedy, spectral, refined):
        assert sol.edges == g.edges_within(sol.nodes)


def test_greedy_plus_plus_converges():
    g = GRAPHS[3]
    assert greedy_plus_plus(g, 200).density == pytest.approx(brute_force(g), rel=1e-3)


@pytest.mark.parametrize("g", GRAPHS[:10])
def test_sweep_dominates_rounded_thresholds(g):
    """The full sweep tries every threshold set, so it is never worse than a subset of thresholds."""
    assert spectral_charikar(g).density >= spectral_charikar(g, decimals=1).density - 1e-12


def test_best_prefix_matches_naive_sweep():
    g = random_graph(7, n=30, p=0.2)
    order = np.random.default_rng(0).permutation(g.n)
    naive = max(g.density(order[: k + 1]) for k in range(g.n))
    assert best_prefix(g, order).density == pytest.approx(naive)


def test_planted_clique_is_found():
    """A 12-clique planted in a sparse random graph is the densest subgraph."""
    rng = np.random.default_rng(1)
    u, v = np.triu_indices(300, 1)
    keep = rng.uniform(size=u.size) < 0.01
    cu, cv = np.triu_indices(12, 1)
    g = Graph.from_edges(np.r_[u[keep], cu], np.r_[v[keep], cv])
    for sol in (goldberg(g), greedy_peeling(g), spectral_refined(g)):
        assert set(range(12)) <= set(sol.nodes.tolist())
        assert sol.density >= 66 / 12


def test_edge_cases():
    single = Graph.from_edges(np.array([0]), np.array([1]))
    for sol in (goldberg(single), charikar_lp(single)[0], greedy_peeling(single), spectral_refined(single)):
        assert sol.density == pytest.approx(0.5)
    assert Solution(np.empty(0, np.int64), 0).density == 0.0
