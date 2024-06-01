from dataclasses import dataclass
from typing import Self

import numpy as np
import scipy.optimize as opt


@dataclass
class ProblemInst:
    s: int
    t: int
    c: np.ndarray
    A_ub: np.ndarray
    b_ub: np.ndarray
    A_eq: np.ndarray
    b_eq: np.ndarray
    sol: np.ndarray

    def get_node_features(self) -> np.ndarray:
        return np.concatenate((self.c, np.dot(self.A_ub, self.c), np.dot(self.A_eq, self.c))).reshape(-1, 1)

    def get_edge_index_and_features(self) -> tuple[np.ndarray, np.ndarray]:
        n_ub_constraints, n_nodes = self.A_ub.shape
        n_eq_constraints, _ = self.A_eq.shape

        edge_index_a = []
        edge_index_b = []

        edge_attr = []

        ub_edges = np.where(self.A_ub > 0)
        edge_index_a.append(ub_edges[0] + n_nodes)
        edge_index_b.append(ub_edges[1])
        edge_index_a.append(ub_edges[1])
        edge_index_b.append(ub_edges[0] + n_nodes)
        edge_attr.append(self.A_ub[ub_edges])
        edge_attr.append(self.A_ub[ub_edges])

        eq_edges = np.where(self.A_eq > 0)
        edge_index_a.append(eq_edges[0] + n_nodes + n_ub_constraints)
        edge_index_b.append(eq_edges[1])
        edge_index_a.append(eq_edges[1])
        edge_index_b.append(eq_edges[0] + n_nodes + n_ub_constraints)
        edge_attr.append(self.A_eq[eq_edges])
        edge_attr.append(self.A_eq[eq_edges])

        edge_index = np.stack([np.concatenate(edge_index_a), np.concatenate(edge_index_b)])
        edge_attr = np.concatenate(edge_attr).reshape(-1, 1)

        return edge_index, edge_attr

    def apply_branched_vars(self, branched_vars: dict[int, int]) -> 'Self':
        A_eq_new = np.copy(self.A_eq)
        b_eq_new = np.copy(self.b_eq)
        for var, value in branched_vars.items():
            A_eq_new[var, var] = 1
            b_eq_new[var] = 1

        return ProblemInst(self.s, self.t, self.c, self.A_ub, self.b_ub, A_eq_new, b_eq_new, self.sol)

    def solve_lp(self, integrality: int = 0) -> opt.OptimizeResult:
        return opt.linprog(self.c, A_ub=self.A_ub, b_ub=self.b_ub, A_eq=self.A_eq, b_eq=self.b_eq, method='highs',
                           integrality=integrality, bounds=(0, 1))
