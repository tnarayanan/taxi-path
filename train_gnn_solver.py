import copy

import numpy as np
import torch.optim
from tqdm import tqdm

from data.airport_graph import AirportGraph
from data.problem_inst import ProblemInst
from gnn_solver import GNNSolver
from model import GNNSolverModel


def get_prob(n_vars: int = 3) -> ProblemInst:
    rng = np.random.default_rng()
    solvable = False
    inst = None
    output = None

    # print("Finding problem... ", end="")
    while not solvable:
        n_cons = 10
        A_ub = np.zeros((n_cons, n_vars))
        b_ub = np.zeros(n_cons)

        for k in range(n_cons):
            x, y = rng.integers(0, n_vars, size=2)
            A_ub[k, x], A_ub[k, y], b_ub[k] = rng.integers(-20, 20, size=3)

        A_eq = np.zeros((n_vars, n_vars))
        b_eq = np.zeros(n_vars)

        # A_eq[0, 0], b_eq[0] = 1, rand_nums[0]
        # A_eq[1, 1], A_eq[1, 2], b_eq[1] = 1, 1, rand_nums[0] - rand_nums[1]
        # A_eq[2, 1], A_eq[2, 2], b_eq[1] = -1, 1, rand_nums[0] + rand_nums[1]

        c = rng.integers(-20, 20, size=n_vars)

        inst = ProblemInst(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, sol=np.zeros(n_vars))

        output = inst.solve_lp(integrality=1)

        solvable = output.success

    inst.sol = output.x
    # print("found problem")
    # print(inst)
    return inst


def main():
    model = GNNSolverModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    solver = GNNSolver(model)

    num_minibatches = 8
    minibatch_size = 16
    dataset = [get_prob(10) for _ in range(num_minibatches * minibatch_size)]
    test_dataset = [get_prob(10) for _ in range(16)]
    losses = []
    test_losses = []

    for i in range(30):
        optimizer.zero_grad()
        for split in tqdm(range(num_minibatches)):
            minibatch_loss = None
            for prob in dataset[split * minibatch_size:(split + 1) * minibatch_size]:
                prob_copy = copy.deepcopy(prob)
                _, _, loss = solver.solve(prob_copy, training=True, print_output=False)
                if loss is not None:
                    minibatch_loss = loss if minibatch_loss is None else minibatch_loss + loss

            if minibatch_loss is not None:
                losses.append(minibatch_loss.item())
                minibatch_loss.backward()
                optimizer.step()

        test_loss = None
        for prob in test_dataset:
            prob_copy = copy.deepcopy(prob)
            _, _, loss = solver.solve(prob_copy, training=True, print_output=False)
            if loss is not None:
                test_loss = loss if test_loss is None else test_loss + loss

        print(f"Epoch {i}: {test_loss.item() if test_loss is not None else "None"}")
        if test_loss is not None:
            test_losses.append(test_loss.item())

    print(losses)
    print(test_losses)

    solver.solve(AirportGraph.KDCA().get_random_problem_inst())


if __name__ == '__main__':
    main()
