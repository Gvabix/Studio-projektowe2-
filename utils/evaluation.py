import sys
import os
import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve

import numpy as np

# === Ścieżki ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "hpnf")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "models", "checkpoints", "hpnf.pt")
os.makedirs(RESULTS_DIR, exist_ok=True)

# === Urządzenie ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Importy lokalne ===
sys.path.append(BASE_DIR)
from models.hpnf import HPNF
from datasets.graph_dataset import GraphDataset


# === Funkcje pomocnicze ===
def load_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def plot_confusion_matrix(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            preds = out.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=np.unique(all_labels),
                yticklabels=np.unique(all_labels))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()


def classification_report_metrics(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            preds = out.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    report = classification_report(all_labels, all_preds, target_names=["Fake", "Real"])
    print(report)
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(report)


def plot_roc_curve(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            all_preds.append(out.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    if all_preds.ndim == 2:
        all_preds = all_preds[:, 1]  # Prob. klasy 1

    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"))
    plt.close()


def plot_precision_recall_curve(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            all_preds.append(out.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    if all_preds.ndim == 2:
        all_preds = all_preds[:, 1]

    precision, recall, _ = precision_recall_curve(all_labels, all_preds)

    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "precision_recall_curve.png"))
    plt.close()


# === Model i dane ===
model = HPNF(in_channels=5, hidden_channels=64).to(device)
model = load_model(model, CHECKPOINT_PATH)

dataset = GraphDataset(root=os.path.join(BASE_DIR, "data"))
dataset = dataset.shuffle()

split_idx = int(0.8 * len(dataset))
val_dataset = dataset[split_idx:]
val_loader = DataLoader(val_dataset, batch_size=32)

# === Generowanie wyników ===
plot_confusion_matrix(model, val_loader)
classification_report_metrics(model, val_loader)
plot_roc_curve(model, val_loader)
plot_precision_recall_curve(model, val_loader)
