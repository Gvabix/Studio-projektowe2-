import os
import torch
from torch_geometric.loader import DataLoader
from torch.nn import functional as F
from sklearn.metrics import accuracy_score
from datasets.graph_dataset import GraphDataset
from models.hpnf import HPNF


# Trening modelu
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# Testowanie modelu
def test(model, loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    return accuracy_score(y_true, y_pred)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset
dataset = GraphDataset(root='../data')
dataset = dataset.shuffle()
split_idx = int(0.8 * len(dataset))
train_dataset = dataset[:split_idx]
test_dataset = dataset[split_idx:]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Model
sample = dataset[0]
in_channels = sample.x.size(1)
model = HPNF(in_channels=in_channels, hidden_channels=64).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(1, 21):
    loss = train(model, train_loader, optimizer, criterion)
    acc = test(model, test_loader)
    print(f"Epoch {epoch:02d}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')
# Zapis modelu po treningu
torch.save(model.state_dict(), 'checkpoints/hpnf.pt')


# Aktualnie za każdym razem tworzony i trenowany jest nowy model.
# Na przyszłość można zastosować wczytywanie zapisanego modelu do dalszego treningu lub testowania:

# import argparse
# import os
#
# parser = argparse.ArgumentParser()
# parser.add_argument("--mode", choices=["train", "test", "resume"], default="train")
# args = parser.parse_args()
#
# model = HPNF(...).to(device)
# model_path = 'checkpoints/hpnf.pt'
#
# if args.mode in ["resume", "test"] and os.path.exists(model_path):
#     print("Ładowanie zapisanych wag...")
#     model.load_state_dict(torch.load(model_path))
# else:
#     print("Trening nowego modelu...")
#
# if args.mode in ["train", "resume"]:
#     torch.save(model.state_dict(), model_path)
#
#
# # użycie:
# python models/train_hpnf.py --mode train   # uczy od nowa i zapisuje
# python models/train_hpnf.py --mode resume  # kontynuuje
# python models/train_hpnf.py --mode test    # testuje bez treningu
