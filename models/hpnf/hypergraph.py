import torch
import torch.nn as nn
import torch.nn.functional as F

class HypergraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, aggregation_type='sum'):
        super(HypergraphConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregation_type = aggregation_type
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels))  # learnable weights

    def forward(self, x, hyperedges):
        # x: node features [num_nodes, in_channels]
        # hyperedges: list of hyperedges where each hyperedge is a list of node indices
        agg_features = torch.zeros_like(x)
        
        for hyperedge in hyperedges:
            # Aggregate features for nodes in the same hyperedge
            nodes_in_edge = x[hyperedge]
            if self.aggregation_type == 'sum':
                agg_features[hyperedge] += nodes_in_edge.sum(dim=0)
            elif self.aggregation_type == 'mean':
                agg_features[hyperedge] += nodes_in_edge.mean(dim=0)
        
        # Apply learnable weights to aggregated features
        out = F.relu(torch.matmul(agg_features, self.weight))
        return out
