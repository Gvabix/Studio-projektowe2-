import torch
import networkx as nx
import numpy as np


def build_edge_index(edges):
    # edges: list of (source, target)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def build_node_features(nodes):
    features = []

    for node in nodes:
        type_feat = [0, 0, 0]
        if node["type"] in [1, 2, 3]:
            type_feat[node["type"] - 1] = 1

        time = node["time"] if node["time"] is not None else 0
        time = [time / 1e10]  # normalize

        bot_score = [node.get("bot_score", 0) or 0]  # None → 0

        # concat all features
        node_feat = type_feat + time + bot_score
        features.append(node_feat)

    return torch.tensor(features, dtype=torch.float)
