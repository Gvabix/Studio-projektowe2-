import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from datasets.hybrid_graph_dataset import HybridGraphDataset
from models.hpnf_rst.hpnf_rst import HPNF_RST

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "hpnf_rst")
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_model(model_path, device, in_channels_graph, rst_channels=27, hidden_channels=64):
    model = HPNF_RST(in_channels_graph, rst_channels, hidden_channels)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def plot_and_save_metrics(y_true, y_pred, y_prob, results_dir, label_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()

    report = classification_report(y_true, y_pred, target_names=label_names)
    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    print("=== Classification Report ===")
    print(report)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0,1],[0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "roc_curve.png"))
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "precision_recall_curve.png"))
    plt.close()

def evaluate(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = HybridGraphDataset(root=os.path.join(BASE_DIR, 'data'), rst_feature_path=os.path.join(BASE_DIR, 'data', 'id_rst_dataset.pkl'))

    from sklearn.model_selection import train_test_split
    labels = [data.y.item() for data in dataset]
    _, test_indices = train_test_split(list(range(len(dataset))), test_size=0.2, stratify=labels, random_state=42)
    test_dataset = [dataset[i] for i in test_indices]

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    sample = dataset[0]
    in_channels_graph = sample.x.size(1)

    model = load_model(model_path, device, in_channels_graph)

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            probs = F.softmax(out, dim=1)[:, 1]
            preds = out.argmax(dim=1)
            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    plot_and_save_metrics(y_true, y_pred, y_prob, RESULTS_DIR, label_names=["Fake", "Real"])

if __name__ == "__main__":
    model_path = os.path.join(BASE_DIR, "models", "hpnf_rst", "checkpoints", "hpnf_rst.pt")
    print(f"Loading model from: {model_path}")
    evaluate(model_path)

