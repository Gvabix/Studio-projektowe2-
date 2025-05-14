import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class HPNF(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(HPNF, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.dropout = 0.5
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)  # Nowa warstwa
        self.classifier = nn.Linear(hidden_channels // 2, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.relu(self.fc1(x))  # Nowa warstwa
        out = self.classifier(x)
        return out
