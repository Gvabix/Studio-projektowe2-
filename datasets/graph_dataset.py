import os
import torch
from torch_geometric.data import InMemoryDataset, Data
from utils.json_parser import parse_graph_json
from utils.build_graph import build_node_features, build_edge_index
import torch.serialization


class GraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)

        # Załaduj dane bez używania weights_only=True
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        # Folder z JSON-ami fake i real
        return []

    @property
    def processed_file_names(self):
        return ['graph_data.pt']

    def process(self):
        data_list = []

        base_path = os.path.join(self.root, 'graphs')
        sources = ['politifact', 'gossipcop']
        labels = {'fake': 0, 'real': 1}

        for source in sources:
            for label_name, label_val in labels.items():
                dir_path = os.path.join(base_path, source, label_name)

                if not os.path.exists(dir_path):
                    continue  # dla bezpieczeństwa

                for file_name in os.listdir(dir_path):
                    if not file_name.endswith('.json'):
                        continue

                    json_path = os.path.join(dir_path, file_name)

                    try:
                        # Parsowanie grafu z JSON-a
                        nodes, edges = parse_graph_json(json_path)

                        # Sprawdzenie, czy wszystkie indeksy w edge_index są w zakresie 0 <= index < len(nodes)
                        edge_index = build_edge_index(edges)
                        num_nodes = len(nodes)
                        edge_index = edge_index[:, edge_index[0] < num_nodes]
                        edge_index = edge_index[:, edge_index[1] < num_nodes]

                        # Budowanie cech węzłów
                        x = build_node_features(nodes)

                        # Tworzenie obiektu Data dla grafu
                        data = Data(x=x, edge_index=edge_index, y=torch.tensor([label_val]))
                        data_list.append(data)
                    except Exception as e:
                        print(f"Błąd w pliku {json_path}: {e}")

        print(f"Przetworzono {len(data_list)} grafów.")
        data, slices = self.collate(data_list)

        # Zapisz dane do pliku .pt
        torch.save((data, slices), self.processed_paths[0])

