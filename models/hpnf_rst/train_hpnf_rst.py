import os
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from datasets.hybrid_graph_dataset import HybridGraphDataset
from models.hpnf_rst.hpnf_rst import HPNF_RST
from torch.nn import functional as F
from torch.optim import Adam
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.75):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - p_t)**self.gamma * BCE_loss
        return focal_loss.mean()

def train(model, loader, optimizer, criterion, device):
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

from sklearn.metrics import precision_score, recall_score

def test(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return acc, f1, precision, recall

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = HybridGraphDataset(root='../../data', rst_feature_path='../../data/id_rst_dataset.pkl')
    dataset = dataset.shuffle()

    from sklearn.model_selection import train_test_split

    labels = [data.y.item() for data in dataset]

    train_indices, test_indices = train_test_split(
        list(range(len(dataset))),
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    train_dataset = [dataset[i] for i in train_indices]
    test_dataset = [dataset[i] for i in test_indices]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    sample = dataset[0]
    in_channels_graph = sample.x.size(1)

    model = HPNF_RST(in_channels_graph, 27, hidden_channels=64).to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = FocalLoss(gamma=2, alpha=0.75)

    for epoch in range(1, 21):
        loss = train(model, train_loader, optimizer, criterion, device)
        acc, f1, precision, recall = test(model, test_loader, device)
        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

    plot_roc(model, test_loader, device)

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    torch.save(model.state_dict(), 'checkpoints/hpnf_rst.pt')

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc(model, loader, device):
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            probs = F.softmax(out, dim=1)[:, 1]
            y_true.extend(data.y.cpu().numpy())
            y_score.extend(probs.cpu().numpy())

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    main()
