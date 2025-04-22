import os
import sys
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score

# Add root directory to sys.path for easier imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.graph_dataset import GraphDataset
from models.hpnf import HPNF

# Set device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, train_loader, val_loader, optimizer, criterion, scheduler=None):
    """Train the model for one epoch and return training and validation loss."""
    model.train()
    total_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Update scheduler if provided
    if scheduler:
        scheduler.step(total_loss)

    # Compute validation loss
    val_loss = evaluate(model, val_loader, criterion)

    return total_loss / len(train_loader), val_loss


def evaluate(model, loader, criterion):
    """Evaluate the model and return average loss."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item()

    return total_loss / len(loader)


def early_stopping(train_loader, val_loader, model, optimizer, criterion, scheduler=None, patience=5, num_epochs=50):
    """Train with early stopping and return the best model."""
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(1, num_epochs + 1):
        train_loss, val_loss = train(model, train_loader, val_loader, optimizer, criterion, scheduler)

        print(f"Epoch {epoch:02d}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    model.load_state_dict(best_model_state)
    print("\nFinal Evaluation on Validation Set:")
    final_val_loss = evaluate(model, val_loader, criterion)
    print(f"Final Validation Loss: {final_val_loss:.4f}")

    return model


def fine_tune(model, train_loader, val_loader, checkpoint_path, optimizer_type='adam', learning_rate=0.0001,
              num_epochs=20, patience=5, use_scheduler=True, scheduler_patience=2, scheduler_factor=0.5):
    """
    Fine-tune a pre-trained model with early stopping and optional learning rate scheduler.
    """
    model.load_state_dict(torch.load(checkpoint_path))
    model = model.to(device)

    # Select optimizer
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    # Initialize learning rate scheduler if needed
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=scheduler_patience, factor=scheduler_factor) if use_scheduler else None

    criterion = F.cross_entropy
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = evaluate(model, val_loader, criterion)

        print(f"Epoch {epoch:02d}, Train Loss: {total_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if scheduler:
            scheduler.step(val_loss)

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)

    print("\nFinal Evaluation on Validation Set:")
    final_val_loss = evaluate(model, val_loader, criterion)
    print(f"Final Validation Loss: {final_val_loss:.4f}")

    save_path = os.path.join('checkpoints', 'tmp.pt')
    torch.save(model.state_dict(), save_path)
    print(f"Fine-tuned model saved to {save_path}")

    return model


if __name__ == '__main__':
    # Load dataset
    dataset = GraphDataset(root='data').shuffle()
    split_idx = int(0.8 * len(dataset))
    train_dataset = dataset[:split_idx]
    val_dataset = dataset[split_idx:]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Initialize model
    model = HPNF(in_channels=dataset.num_features, hidden_channels=64).to(device)

    # Fine-tune
    fine_tuned_model = fine_tune(model, train_loader, val_loader, checkpoint_path='checkpoints/hpnf_best.pt',
                                 optimizer_type='adam', learning_rate=0.0001, num_epochs=20, patience=5, use_scheduler=True)

    # Usage:
    # python models/train_hpnf.py --mode train   # trains from scratch
    # python models/train_hpnf.py --mode resume  # resumes from checkpoint
