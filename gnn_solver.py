from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch

from data.airport_graph import AirportGraph
from data.problem_inst import ProblemInst
from model import GNNSolverModel


@dataclass
class GNNSolverState:
    problem_inst: ProblemInst
    branched_vars: dict[int, int]

class GNNSolver:
    def __init__(self, model: GNNSolverModel):
        self.model = model

    def solve(self, inst: ProblemInst):
        self.model.eval()
        n = inst.c.shape[0]

        x_star = np.zeros(inst.c.shape)
        z_star = np.inf

        stack: list[GNNSolverState] = [GNNSolverState(problem_inst=inst, branched_vars={})]

        while len(stack) > 0:
            state = stack.pop()
            inst = state.problem_inst
            branched_vars = state.branched_vars

            # first solve the LP relaxation of this problem inst
            solution = inst.solve_lp()

            # skip problem if infeasible
            if not solution.success:
                continue

            # get solution and objective value
            x_j = solution.x
            z_j = np.dot(inst.c, x_j)

            is_solution_integral = np.sum(np.isclose(x_j, np.ones(n)) + np.isclose(x_j, np.zeros(n))) == n

            if not is_solution_integral:
                print(f"LP relaxation: {z_j}, {x_j}, {is_solution_integral=}")

            if z_j < z_star and not is_solution_integral:
                # determine which variable to branch on
                node_features = inst.get_node_features()
                print(inst.c.shape, node_features.shape)

                edge_index, edge_attr = inst.get_edge_index_and_features()
                print(edge_index.shape, edge_attr.shape)

                model_outputs: torch.Tensor = self.model(torch.from_numpy(node_features).type(torch.FloatTensor),
                                                         torch.from_numpy(edge_index),
                                                         torch.from_numpy(edge_attr).type(torch.FloatTensor))[:n]
                print(model_outputs.shape)

                model_outputs[np.array(list(state.branched_vars.keys()))] = -torch.inf

                selected_var = model_outputs.argmax(dim=0).item()
                print(selected_var)

                branched_1 = deepcopy(branched_vars)
                branched_1[selected_var] = 1

                branched_0 = deepcopy(branched_vars)
                branched_0[selected_var] = 1

                state_1 = GNNSolverState(problem_inst=inst.apply_branched_vars(branched_1), branched_vars=branched_1)
                state_0 = GNNSolverState(problem_inst=inst.apply_branched_vars(branched_0), branched_vars=branched_0)

                stack.extend([state_1, state_0])
            elif z_j < z_star and is_solution_integral:
                x_star, z_star = x_j, z_j
                print(f"Found better solution with value {z_star}: {x_star}")

        print()
        print(f"branch and bound solution: {z_star}, {x_star}")

        optimal_sol = inst.solve_lp(integrality=1)
        print(f"optimal solution: {np.dot(inst.c, optimal_sol.x)}, {optimal_sol.x}")

        return x_star, z_star


if __name__ == '__main__':
    GNNSolver(GNNSolverModel()).solve(AirportGraph.KDCA().to_problem_inst())