import os
import torch
from torch_geometric.data import InMemoryDataset, Data
from utils.json_parser import parse_graph_json
from utils.build_graph import build_node_features, build_edge_index
import pandas as pd
import numpy as np

class HybridGraphDataset(InMemoryDataset):
    def __init__(self, root, rst_feature_path, transform=None, pre_transform=None):
        self.rst_feature_path = rst_feature_path
        super().__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)


    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['hybrid_graph_data.pt']

    def process(self):
        print("Wczytywanie cech RST...")
        df = pd.read_pickle(self.rst_feature_path)
        rst_features = {
            row["id"]: row.drop("id").to_numpy(dtype=np.float32)
            for _, row in df.iterrows()
        }

        data_list = []
        base_path = os.path.join(self.root, 'filtered', 'graphs_filtered')
        sources = ['politifact', 'gossipcop']
        labels = {'fake': 0, 'real': 1}

        for source in sources:
            for label_name, label_val in labels.items():
                dir_path = os.path.join(base_path, source, label_name)

                if not os.path.exists(dir_path):
                    continue

                for file_name in os.listdir(dir_path):
                    if not file_name.endswith('.json'):
                        continue

                    json_path = os.path.join(dir_path, file_name)
                    graph_id = file_name.replace('.json', '')

                    if graph_id not in rst_features:
                        print(f"Brak cech RST dla: {graph_id}")
                        continue

                    try:
                        nodes, edges = parse_graph_json(json_path)
                        edge_index = build_edge_index(edges)
                        num_nodes = len(nodes)

                        edge_index = edge_index[:, edge_index[0] < num_nodes]
                        edge_index = edge_index[:, edge_index[1] < num_nodes]

                        x = build_node_features(nodes)

                        degrees = [len([e for e in edges if e[0] == i or e[1] == i]) for i in range(num_nodes)]
                        x = torch.cat([x, torch.tensor(degrees).view(-1, 1)], dim=1)

                        rst_feat = torch.tensor(rst_features[graph_id]).float().unsqueeze(0)

                        data = Data(
                            x=x,
                            edge_index=edge_index,
                            y=torch.tensor([label_val]),
                            rst=rst_feat,
                            id=graph_id
                        )

                        data_list.append(data)
                    except Exception as e:
                        print(f"Błąd w pliku {json_path}: {e}")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
