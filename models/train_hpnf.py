import os
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from datasets.graph_dataset import GraphDataset
from models.hpnf import HPNF
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.75):  # Zmiana alpha!
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - p_t)**self.gamma * BCE_loss
        return focal_loss.mean()

# Trening
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

# Test
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

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return acc, f1

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

optimizer = Adam(model.parameters(), lr=0.001)
criterion = FocalLoss(gamma=2, alpha=0.75)  # <-- tu zmiana

# Training loop
for epoch in range(1, 21):
    loss = train(model, train_loader, optimizer, criterion)
    acc, f1 = test(model, test_loader)
    print(f"Epoch {epoch:02d}, Loss: {loss:.4f}, Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

# Save
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

torch.save(model.state_dict(), 'checkpoints/hpnf.pt')
