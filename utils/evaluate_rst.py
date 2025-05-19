import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve

# === Paths and Imports ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "datasets"))


RESULTS_DIR = os.path.join(BASE_DIR, "results", "rst")
os.makedirs(RESULTS_DIR, exist_ok=True)

# === Metrics Utility ===
def plot_and_save_metrics(preds, probs, labels, results_dir, label_names):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()

    report = classification_report(labels, preds, target_names=label_names)
    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "roc_curve.png"))
    plt.close()

    precision, recall, _ = precision_recall_curve(labels, probs)
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "precision_recall_curve.png"))
    plt.close()

    report = classification_report(labels, preds, target_names=label_names)
    print("=== Classification Report ===")
    print(report)  # <-- Add this line to see output in terminal

    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
        f.write(report)


# === Evaluation ===
def evaluate_rst():
    print("Evaluating RST (Random Forest)...")
    model_path = os.path.join(BASE_DIR, "models", "rst", "results", "rf_model.pth")
    rst_model = torch.load(model_path)

    dataset = pd.read_pickle(os.path.join(BASE_DIR, "data", "rst_dataset.pkl"))
    val_dataset = dataset[int(0.8 * len(dataset)):]

    X = val_dataset.drop(columns=['label'])
    y = val_dataset['label']

    y_pred = rst_model.predict(X)
    y_prob = rst_model.predict_proba(X)[:, 1]

    plot_and_save_metrics(
        y_pred,
        y_prob,
        y,
        RESULTS_DIR,
        label_names=["Fake", "Real"]
    )

if __name__ == "__main__":
    evaluate_rst()
