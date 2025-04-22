import sys
import os
import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import label_binarize

import numpy as np


# Define a function to load the model from a checkpoint
def load_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)  # Load the checkpoint
    model.load_state_dict(checkpoint)  # Load the weights into the model
    model.to(device)  # Ensure the model is on the correct device
    model.eval()  # Set the model to evaluation mode
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
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(all_labels), yticklabels=np.unique(all_labels))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()


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
    
    report = classification_report(all_labels, all_preds, target_names=[f"Class {i}" for i in range(len(np.unique(all_labels)))])
    print(report)

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


    # Binary classification
    if all_preds.ndim == 2:
        all_preds = all_preds[:, 0]  # Flatten to shape [n_samples]
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Binary)')
    plt.legend(loc='lower right')
    plt.grid()
    plt.tight_layout()
    plt.show()
    plt.close()




def plot_loss_curves(train_losses, val_losses):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.show()

def plot_accuracy(train_accuracies, val_accuracies):
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')
    plt.show()



def plot_precision_recall_curve(model, loader, num_classes=2):
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


    # Binary classification case
    if all_preds.ndim == 2:
        all_preds = all_preds[:, 0]
    precision, recall, _ = precision_recall_curve(all_labels, all_preds)

    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid()
    plt.tight_layout()
    plt.show()
    plt.close()


 # Load model and data
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.hpnf import HPNF
from datasets.graph_dataset import GraphDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure the model's configuration is the same as during training
in_channels = 5  # or the correct value used during training
hidden_channels = 64  # the correct value for hidden channels
model = HPNF(in_channels=in_channels, hidden_channels=hidden_channels).to(device)
 # Replace with your model's architecture and parameters
model = load_model(model, 'checkpoints/fine_tuned_model.pt')  # Load the saved fine-tuned model

# Load your dataset
dataset = GraphDataset(root='data')
dataset = dataset.shuffle()

# Split the dataset (80% train, 20% validation)
split_idx = int(0.8 * len(dataset))
train_dataset = dataset[:split_idx]
val_dataset = dataset[split_idx:]

# Define dataloaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)





# plot_confusion_matrix(model, val_loader)
# classification_report_metrics(model, val_loader)
# plot_roc_curve(model, val_loader)

train_losses = [0.5589, 0.5589, 0.5563, 0.5587, 0.5563, 0.5584]  # Store train losses for each epoch
val_losses = [0.5241, 0.5251, 0.5248, 0.5251, 0.5250, 0.5251]  # Store validation losses for each epoch

# train_losses.append(train_loss)
# val_losses.append(val_loss)

# plot_loss_curves(train_losses, val_losses)


plot_precision_recall_curve(model, val_loader, num_classes=2)
