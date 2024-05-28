# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch_geometric.nn as gnn

from data.airport_graph import AirportGraph
from data.problem_inst import ProblemInst

x = AirportGraph.KDCA()

print(x.adj_list)

print(ProblemInst.from_airport_graph(x))
