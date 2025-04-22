import sys
import os

# Adds the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph_dataset import GraphDataset


def main():
    print("Przetwarzanie danych grafowych...")
    dataset = GraphDataset(root='data')
    print(f"Zapisano {len(dataset)} przykładów do: data/processed/graph_data.pt")


if __name__ == "__main__":
    main()
