import torch
from torch import nn

from .utils import ConvBlock


class FCNBaseline(nn.Module):
    """
    Fully Convolutional Network baseline for time-series classification.
    """

    def __init__(self, in_channels: int, num_pred_classes: int = 1) -> None:
        super().__init__()

        # Stored for easy model loading
        self.input_args = {
            "in_channels": in_channels,
            "num_pred_classes": num_pred_classes
        }

        # Convolutional feature extractor
        self.layers = nn.Sequential(
            ConvBlock(in_channels, 128, 8, 1),
            ConvBlock(128, 256, 5, 1),
            ConvBlock(256, 128, 3, 1),
        )

        # Final classifier
        self.final = nn.Linear(128, num_pred_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, channels, sequence_length)
        """

        # Extract temporal features
        x = self.layers(x)

        # Global Average Pooling over time dimension
        x = x.mean(dim=-1)

        # Classification
        return self.final(x)
