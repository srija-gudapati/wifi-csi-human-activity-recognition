import logging
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dataset import CSIDataset
from model import CSIModel   # your model class


# -------------------------
# Setup
# -------------------------
logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

BATCH_SIZE = 16
EPOCHS = 40
LR = 1e-3


# -------------------------
# Load Dataset
# -------------------------
def load_data():
    dataset = CSIDataset([
        "dataset/bedroom_lviv/1"
    ])

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader


# -------------------------
# Training step
# -------------------------
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for x, y in tqdm(loader, desc="Training"):
        x = x.float().to(device)
        y = y.long().to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# -------------------------
# Validation step
# -------------------------
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validation"):
            x = x.float().to(device)
            y = y.long().to(device)

            out = model(x)
            loss = criterion(out, y)

            total_loss += loss.item()

            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total
    return total_loss / len(loader), acc


# -------------------------
# Main training loop
# -------------------------
def train():
    train_loader, val_loader = load_data()

    model = CSIModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_acc = 0

    for epoch in range(EPOCHS):
        logging.info(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        scheduler.step(val_loss)

        logging.info(f"Train Loss: {train_loss:.4f}")
        logging.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "model.pth")
            logging.info("Best model saved.")

    logging.info(f"\nFinal Best Accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    train()
