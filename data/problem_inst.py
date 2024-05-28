from dataclasses import dataclass
from typing import Self

import numpy as np

from data.airport_graph import AirportGraph


@dataclass
class ProblemInst:
    airport_graph: AirportGraph
    s: int
    t: int
    c: np.ndarray
    A_ub: np.ndarray
    b_ub: np.ndarray
    A_eq: np.ndarray
    b_eq: np.ndarray
    sol: np.ndarray

    @classmethod
    def from_airport_graph(cls, airport_graph: AirportGraph) -> 'Self':
        rng = np.random.default_rng()
        n = airport_graph.n_intersection_nodes + airport_graph.n_taxiway_nodes

        # select random pairs of nodes to be claimed
        num_planes_on_tarmac = rng.integers(1, airport_graph.n_intersection_nodes // 2)
        all_nodes = set(range(n))
        claimed_nodes = set()
        while len(claimed_nodes) < 2 * num_planes_on_tarmac:
            node1 = rng.choice(list(all_nodes - claimed_nodes))
            available_neighbors = list(airport_graph.adj_list[node1] - claimed_nodes)
            if len(available_neighbors) == 0:
                continue
            node2 = rng.choice(available_neighbors)
            claimed_nodes.update([node1, node2])

        # look for s and t such that they are connected
        path = []
        s, t = -1, -1
        while len(path) == 0:
            s, t = rng.choice(list(set(range(airport_graph.n_intersection_nodes)) - claimed_nodes), size=2,
                              replace=False)
            path, _ = airport_graph.shortest_path(s, t, restricted_nodes=claimed_nodes)

        # construct problem matrices

        # lengths for objective function
        c = np.array([0 for _ in range(airport_graph.n_intersection_nodes)] + airport_graph.taxiway_lens)

        # empty equality constraint matrices
        A_eq = np.zeros((n + 3, n))
        b_eq = np.zeros(n + 3)

        # y_s = 1
        A_eq[0, s] = 1
        b_eq[0] = 1

        # y_t = 1
        A_eq[1, s] = 1
        b_eq[1] = 1

        # sum (c_i * y_i) = 0
        for nd in claimed_nodes:
            A_eq[2, nd] = 1
        b_eq[2] = 0

        # sum_(s,j) y_j = 1
        for nd in airport_graph.adj_list[s]:
            A_eq[3, nd] = 1
        b_eq[3] = 1

        # sum_(t,j) y_j = 1
        for nd in airport_graph.adj_list[t]:
            A_eq[4, nd] = 1
        b_eq[4] = 1

        A_ub = np.zeros((2 * (n - 2), n))
        b_ub = np.zeros(2 * (n - 2))

        # sum_(i,j) y_j - (2-M)*y_i <= M
        # sum_(i,j) y_j - (2+M)*y_i >= -M    ==>    sum_(i,j) -y_j + (2+M)*y_i <= M
        m = 10
        for idx, i in enumerate(all_nodes - {s, t}):
            for j in airport_graph.adj_list[i]:
                A_ub[2*idx, j] = 1
                A_ub[2*idx + 1, j] = -1

            A_ub[2*idx, i] = -(2 - m)
            A_ub[2*idx + 1, i] = 2 + m

            b_ub[2*idx] = m
            b_ub[2*idx + 1] = m

        sol = np.zeros(n)
        for nd in path:
            sol[nd] = 1

        return ProblemInst(airport_graph, s, t, c, A_ub, b_ub, A_eq, b_eq, sol)
