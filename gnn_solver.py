import numpy as np
import torch
import torch.nn.functional as F

from data.airport_graph import AirportGraph
from data.problem_inst import ProblemInst
from model import GNNSolverModel


class GNNSolver:
    def __init__(self, model: GNNSolverModel):
        self.model = model

    def solve(self, inst: ProblemInst, training: bool = False, print_output: bool = True):
        if training:
            self.model.train()
        else:
            self.model.eval()

        n = inst.c.shape[0]

        x_star = np.zeros(inst.c.shape)
        z_star = np.inf

        stack: list[ProblemInst] = [inst]

        loss = None

        while len(stack) > 0:
            inst = stack.pop()

            # first solve the LP relaxation of this problem inst
            solution = inst.solve_lp()

            # skip problem if infeasible
            if not solution.success:
                continue

            # get solution and objective value
            x_j = solution.x
            z_j = np.dot(inst.c, x_j)

            is_solution_integral = np.sum(np.isclose(x_j, np.ones(n)) + np.isclose(x_j, np.zeros(n))) == n

            if not is_solution_integral and print_output:
                print(f"LP relaxation: {z_j}, {x_j}, {is_solution_integral=}")

            if z_j < z_star and not is_solution_integral:
                vars_already_branched = inst.branched_vars.keys()

                # calculate optimal variable to branch on
                best = (-1, -9999999)
                for i in set(range(n)) - vars_already_branched:
                    b0, b1 = inst.branch(i, 0), inst.branch(i, 1)
                    opt0, opt1 = b0.solve_lp(), b1.solve_lp()
                    if opt0.success and opt1.success and (fsb_val := (z_j - opt0.fun) * (z_j - opt1.fun)) > best[1]:
                        best = (i, fsb_val)

                if best[0] == -1:
                    best = ((set(range(n)) - vars_already_branched).pop(), -1)
                # print(f"{best=}")

                # use model determine which variable to branch on
                node_features = inst.get_node_features()
                node_features = F.normalize(torch.from_numpy(node_features).type(torch.FloatTensor), dim=1)
                # print(inst.c.shape, node_features.shape)

                edge_index, edge_attr = inst.get_edge_index_and_features()
                edge_index = torch.from_numpy(edge_index)
                edge_attr = F.normalize(torch.from_numpy(edge_attr).type(torch.FloatTensor), dim=0)
                # print(edge_index.shape, edge_attr.shape)

                model_outputs: torch.Tensor = self.model(node_features,
                                                         edge_index,
                                                         edge_attr)[:n, 0]

                # print(f"{node_features[:, 0] = }, {edge_index = }, {edge_attr[:, 0] = }, {model_outputs = }")
                model_outputs[np.array(list(vars_already_branched))] = -1e5

                softmax_logits = F.softmax(model_outputs, dim=0)

                # print(f"{model_outputs = }, {softmax_logits = }")

                if training:
                    try:
                        selected_var = np.random.default_rng().choice(np.arange(softmax_logits.size(0)), p=softmax_logits.detach().numpy())
                    except ValueError as e:
                        print(f"{model_outputs = }, {softmax_logits = }")
                        raise e
                else:
                    selected_var = softmax_logits.argmax(dim=0).item()
                # print(selected_var, vars_already_branched, model_outputs)

                stack.extend([inst.branch(selected_var, 0), inst.branch(selected_var, 1)])

                cur_loss = -torch.log(softmax_logits[best[0]])

                # print(f"{best=}, {cur_loss=}")
                if loss is None:
                    loss = cur_loss
                else:
                    loss += cur_loss
            elif z_j < z_star and is_solution_integral:
                x_star, z_star = x_j, z_j
                if print_output:
                    print(f"Found better solution with value {z_star}: {x_star}")

        if print_output:
            print()
            print(f"branch and bound solution: {z_star}, {x_star}")

            optimal_sol = inst.solve_lp(integrality=1)
            if optimal_sol.success:
                print(f"optimal solution: {np.dot(inst.c, optimal_sol.x)}, {optimal_sol.x}")
                print(f"optimal and discovered solution close: {np.allclose(x_star, optimal_sol.x)}")
            else:
                print("optimal solver found no solution")

            # print()
            # print(f"actual solution: {np.dot(inst.c, inst.sol)} {inst.sol}")
            # print(f"\tclose to discovered: {np.allclose(x_star, inst.sol)}")
            # if optimal_sol.success:
            #     print(f"\tclose to optimal: {np.allclose(optimal_sol.x, inst.sol)}")

        # print(inst.data['s'], inst.data['t'])
        return x_star, z_star, loss


if __name__ == '__main__':
    GNNSolver(GNNSolverModel()).solve(AirportGraph.KDCA().get_random_problem_inst())
