import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CSIDataset
from model import CSIModel   # <-- your model class

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Validation Function
# -------------------------
def evaluate(model, dl, criterion):
    model.eval()

    correct, total, total_loss = 0, 0, 0

    with torch.no_grad():
        for x_val, y_val in tqdm(dl, desc="Validation", leave=False):
            x_val = x_val.float().to(device)
            y_val = y_val.long().to(device)

            out = model(x_val)
            loss = criterion(out, y_val)

            total_loss += loss.item()

            preds = out.argmax(dim=1)
            total += y_val.size(0)
            correct += (preds == y_val).sum().item()

    acc = correct / total
    return total_loss, acc


# -------------------------
# Training Function
# -------------------------
def train(model, train_dl, val_dl, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for x_train, y_train in tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}"):
            x_train = x_train.float().to(device)
            y_train = y_train.long().to(device)

            optimizer.zero_grad()
            out = model(x_train)
            loss = criterion(out, y_train)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        val_loss, val_acc = evaluate(model, val_dl, criterion)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {running_loss/len(train_dl):.4f}")
        print(f"Val Loss: {val_loss/len(val_dl):.4f}")
        print(f"Val Accuracy: {val_acc*100:.2f}%\n")


# -------------------------
# Main Pipeline
# -------------------------
if __name__ == "__main__":

    # Dataset
    train_dataset = CSIDataset(["dataset/bedroom_lviv/1"])
    val_dataset   = CSIDataset(["dataset/bedroom_lviv/4"])

    train_dl = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dl   = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Model
    model = CSIModel().to(device)

    # Train
    train(model, train_dl, val_dl, epochs=10)
