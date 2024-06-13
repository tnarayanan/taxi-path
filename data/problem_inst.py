from copy import deepcopy
from dataclasses import dataclass, field
from typing import Self, Any

import numpy as np
import scipy.optimize as opt


@dataclass
class ProblemInst:
    c: np.ndarray
    A_ub: np.ndarray
    b_ub: np.ndarray
    A_eq: np.ndarray
    b_eq: np.ndarray
    sol: np.ndarray
    data: dict[Any, Any] = field(default_factory=dict)
    branched_vars: dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        n_eq_constraints, n_vars = self.A_eq.shape
        A_eq_new = np.zeros((n_eq_constraints + n_vars, n_vars))
        b_eq_new = np.zeros(n_eq_constraints + n_vars)

        A_eq_new[n_vars:, :] = self.A_eq
        b_eq_new[n_vars:] = self.b_eq

        for var, value in self.branched_vars.items():
            A_eq_new[var, var] = 1
            b_eq_new[var] = 1

        self.A_eq = A_eq_new
        self.b_eq = b_eq_new

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

    def solve_lp(self, integrality: int = 0) -> opt.OptimizeResult:
        return opt.linprog(self.c, A_ub=self.A_ub, b_ub=self.b_ub, A_eq=self.A_eq, b_eq=self.b_eq, method='highs',
                           integrality=integrality, bounds=(0, 1))

    def branch(self, var: int, value: int) -> 'Self':
        new_branched_vars = deepcopy(self.branched_vars)
        assert var not in new_branched_vars, f"variable {var} has already been branched on (value={new_branched_vars[var]})"
        new_branched_vars[var] = value
        return ProblemInst(self.c, self.A_ub, self.b_ub, self.A_eq, self.b_eq, self.sol, data=self.data, branched_vars=new_branched_vars)
