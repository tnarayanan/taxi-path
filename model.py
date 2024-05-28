from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class IPSolverModel(nn.Module):
    def __init__(self, num_features=1, hidden_size=16, edge_dim=1, target_size=1):
        super().__init__()
        self.convs = [gnn.GATConv(num_features, hidden_size, edge_dim=edge_dim),
                      gnn.GATConv(hidden_size, hidden_size, edge_dim=edge_dim)]
        self.linear = nn.Linear(hidden_size, target_size)

    def forward(self, x, edge_index, edge_attr):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)

        x = self.convs[-1](x, edge_index, edge_attr=edge_attr)
        x = self.linear(x)

        return x
