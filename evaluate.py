import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

from dataset import CSIDataset
from model import CSIModel   # your model class

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Evaluation
# -------------------------
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating"):
            x = x.float().to(device)
            y = y.long().to(device)

            outputs = model(x)
            preds = outputs.argmax(dim=1)

            total += y.size(0)
            correct += (preds == y).sum().item()

    accuracy = correct / total
    print(f"\nFinal Accuracy: {accuracy*100:.2f}%")
    return accuracy


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":

    # Load dataset
    test_dataset = CSIDataset(["dataset/bedroom_lviv/4"])
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Load model
    model = CSIModel().to(device)

    # load trained weights (after training)
    model.load_state_dict(torch.load("model.pth", map_location=device))

    evaluate(model, test_loader)
