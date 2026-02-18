import torch
from torch import nn


class LSTMClassifier(nn.Module):
    """
    Proper LSTM time-series classifier.
    No persistent hidden state.
    Works for train + validation + inference.
    """

    def __init__(self, in_dim, hidden_dim, num_layers, dropout, bidirectional, num_classes):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_dir = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * self.num_dir, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # Fresh hidden state each batch
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * self.num_dir, batch_size, self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers * self.num_dir, batch_size, self.hidden_dim, device=x.device)

        out, _ = self.lstm(x, (h0, c0))

        # Use last timestep
        out = self.classifier(out[:, -1, :])
        return out
