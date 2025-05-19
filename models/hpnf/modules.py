import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import GATConv

class TSPE(nn.Module):
    """Type-specific Path Encoder (TSPE)"""
    def __init__(self, in_channels, hidden_channels):
        super(TSPE, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x

class HSE(nn.Module):
    """Hierarchical Structure Encoder (HSE)"""
    def __init__(self, hidden_channels):
        super(HSE, self).__init__()
        self.attention = GATConv(hidden_channels, hidden_channels, heads=4, dropout=0.5)

    def forward(self, x, data):
        edge_index = data.edge_index
        x = self.attention(x, edge_index)
        return x

class NGE(nn.Module):
    """News Graph Encoder (NGE)"""
    def __init__(self, hidden_channels):
        super(NGE, self).__init__()
        self.fc = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x, data):
        return self.fc(x)
