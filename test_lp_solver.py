import numpy as np
import scipy.optimize as opt

from data.airport_graph import AirportGraph
from data.problem_inst import ProblemInst

x = AirportGraph.KDCA()

inst = ProblemInst.from_airport_graph(x)

res = opt.linprog(inst.c, A_ub=inst.A_ub, b_ub=inst.b_ub, A_eq=inst.A_eq, b_eq=inst.b_eq, method='highs', integrality=1,
                  bounds=(0, 1))

print(inst.s, inst.t)
print(np.where(res.x > 0))
