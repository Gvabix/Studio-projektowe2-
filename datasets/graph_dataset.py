import os
import sys
import torch
from torch_geometric.data import InMemoryDataset, Data
from utils.json_parser import parse_graph_json
from utils.build_graph import build_node_features, build_edge_index

# Adds the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class GraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        self.root = root  # Save root for use in paths
        super().__init__(root, transform, pre_transform)

        if os.path.exists(self.processed_paths[0]):
            print(f"Loading processed data from {self.processed_paths[0]}")
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        else:
            print("Processed data not found. You need to run `process()` first.")

    @property
    def raw_file_names(self):
        return []  # Not used by this dataset

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
                print(f"🔍 Szukam plików w: {dir_path}")

                if not os.path.exists(dir_path):
                    print(f"⚠️ Brak folderu: {dir_path}")
                    continue

                for file_name in os.listdir(dir_path):
                    if not file_name.endswith('.json'):
                        print(f"⛔ Pomijam nie-json: {file_name}")
                        continue

                    json_path = os.path.join(dir_path, file_name)
                    print(f"📄 Przetwarzam: {json_path}")

                    try:
                        nodes, edges = parse_graph_json(json_path)

                        if not nodes:
                            print(f"❌ Brak węzłów w pliku: {json_path}")
                            continue

                        edge_index = build_edge_index(edges)
                        num_nodes = len(nodes)

                        # Correct edge_index filtering
                        edge_index = edge_index[:, (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)]

                        x = build_node_features(nodes)
                        data = Data(x=x, edge_index=edge_index, y=torch.tensor([label_val]))
                        data_list.append(data)

                    except Exception as e:
                        print(f"🛑 Błąd w pliku {json_path}: {e}")

        print(f"✅ Przetworzono {len(data_list)} grafów.")

        if not data_list:
            raise ValueError("❗ Brak danych do zapisania – sprawdź logi powyżej.")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


    # def process(self):
    #     data_list = []

    #     base_path = os.path.join(self.root, 'graphs')
    #     sources = ['politifact', 'gossipcop']
    #     labels = {'fake': 0, 'real': 1}

    #     for source in sources:
    #         for label_name, label_val in labels.items():
    #             dir_path = os.path.join(base_path, source, label_name)
    #             print(f"Looking in: {dir_path}")

    #             if not os.path.exists(dir_path):
    #                 print(f"Directory not found: {dir_path}")
    #                 continue

    #             for file_name in os.listdir(dir_path):
    #                 if not file_name.endswith('.json'):
    #                     continue

    #                 json_path = os.path.join(dir_path, file_name)

    #                 try:
    #                     nodes, edges = parse_graph_json(json_path)

    #                     if not nodes or not edges:
    #                         print(f"Empty graph: {json_path}")
    #                         continue

    #                     edge_index = build_edge_index(edges)
    #                     num_nodes = len(nodes)

    #                     # Filter invalid edges
    #                     mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
    #                     edge_index = edge_index[:, mask]

    #                     x = build_node_features(nodes)

    #                     data = Data(x=x, edge_index=edge_index, y=torch.tensor([label_val]))
    #                     data_list.append(data)
    #                 except Exception as e:
    #                     print(f"Błąd w pliku {json_path}: {e}")

        print(f"✅ Przetworzono {len(data_list)} grafów.")

        if not data_list:
            raise RuntimeError("Brak grafów do zapisania! Upewnij się, że katalogi i pliki są poprawne.")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
