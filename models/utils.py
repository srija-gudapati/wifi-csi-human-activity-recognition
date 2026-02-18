import torch
from torch import nn
import torch.nn.functional as F


class Conv1dSamePadding(nn.Conv1d):
    """
    1D Convolution with TensorFlow-style 'same' padding.
    Keeps output sequence length equal to input length.
    """

    def forward(self, input):
        return conv1d_same_padding(
            input=input,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups
        )


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    """
    Calculates correct padding dynamically.
    Prevents sequence shrinking in deep CNN stacks.
    """

    kernel = weight.size(2)
    stride = stride[0]
    dilation = dilation[0]

    input_length = input.size(2)
    output_length = input_length  # SAME padding target

    padding = ((output_length - 1) * stride - input_length + dilation * (kernel - 1) + 1)

    if padding < 0:
        padding = 0

    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding // 2,
        dilation=dilation,
        groups=groups
    )


class ConvBlock(nn.Module):
    """
    Standard Conv → BatchNorm → ReLU block
    Used by FCN & InceptionTime models
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()

        self.conv = Conv1dSamePadding(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False
        )

        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def _initialize_weights(self):
        """
        Proper initialization prevents exploding gradients in time-series models
        """
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
