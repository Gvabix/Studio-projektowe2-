import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class HPNF_RST(nn.Module):
    def __init__(self, in_channels_graph, in_channels_rst, hidden_channels):
        super(HPNF_RST, self).__init__()
        self.conv1 = GCNConv(in_channels_graph, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.fc_graph = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc_rst = nn.Linear(in_channels_rst, hidden_channels // 2)

        self.classifier = nn.Linear(hidden_channels, 2)
        self.dropout = 0.5

    def forward(self, data):
        x, edge_index, batch, rst = data.x, data.edge_index, data.batch, data.rst

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)

        x_graph = F.relu(self.fc_graph(x))
        x_rst = F.relu(self.fc_rst(rst))

        x_combined = torch.cat([x_graph, x_rst], dim=1)
        out = self.classifier(x_combined)
        return out

