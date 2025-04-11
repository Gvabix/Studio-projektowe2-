import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class HPNF(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.5):
        super(HPNF, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = dropout
        self.classifier = nn.Linear(hidden_channels, 2)  # fake vs real

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        # Poolujemy cały graf do jednej reprezentacji (średnia po węzłach)
        x = global_mean_pool(x, batch)

        out = self.classifier(x)
        return out
