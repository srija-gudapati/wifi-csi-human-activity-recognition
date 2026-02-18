import torch
from torch import nn


class SimpleLSTMClassifier(nn.Module):
    """Clean minimal LSTM classifier"""

    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.0, bidirectional=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_dir = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        self.fc = nn.Linear(hidden_dim * self.num_dir, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device

        h0 = torch.zeros(self.num_layers * self.num_dir, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers * self.num_dir, batch_size, self.hidden_dim, device=device)

        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])
