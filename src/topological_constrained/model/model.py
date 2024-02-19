import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import DataParallel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DownConv(nn.Module):
    """
    Module representing a down-convolutional block.

    This block consists of two convolutional layers with batch normalization and dropout.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        drop_rate (float, optional): Dropout rate (default: 0.2).
        batch_norm_momentum (float, optional): Momentum parameter for batch normalization (default: 0.1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        drop_rate: float = 0.2,
        batch_norm_momentum: float = 0.1,
    ):
        super().__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(out_channels, momentum=batch_norm_momentum)
        self.conv1_drop = nn.Dropout2d(drop_rate)
        init.xavier_normal_(self.conv1.weight)
        init.constant_(self.conv1.bias, 0)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(out_channels, momentum=batch_norm_momentum)
        self.conv2_drop = nn.Dropout2d(drop_rate)
        init.xavier_normal_(self.conv2.weight)
        init.constant_(self.conv2.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the down-convolutional block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # First convolutional layer operations
        x = F.relu(self.conv1(x))
        x = self.conv1_bn(x)
        x = self.conv1_drop(x)

        # Second convolutional layer operations
        x = F.relu(self.conv2(x))
        x = self.conv2_bn(x)
        x = self.conv2_drop(x)

        return x


class UpConv(nn.Module):
    """
    Module representing an up-convolutional block.

    This block consists of an upsampling layer followed by a down-convolutional block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        drop_rate (float, optional): Dropout rate (default: 0.2).
        batch_norm_momentum (float, optional): Momentum parameter for batch normalization (default: 0.1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        drop_rate: float = 0.2,
        batch_norm_momentum: float = 0.1,
    ):
        super().__init__()
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.downconv = DownConv(
            in_channels, out_channels, drop_rate, batch_norm_momentum
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the up-convolutional block.

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Skip connection tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.up1(x)
        x = torch.cat([x, y], dim=1)
        x = self.downconv(x)
        return x


class sUNet(nn.Module):
    """
    Modified U-Net architecture for semantic segmentation.

    This model consists of a downsampling path followed by an upsampling path.

    Args:
        drop_rate (float, optional): Dropout rate (default: 0.2).
        batch_norm_momentum (float, optional): Momentum parameter for batch normalization (default: 0.1).
    """

    def __init__(self, drop_rate: float = 0.2, batch_norm_momentum: float = 0.1):
        super().__init__()

        # Downsampling path
        self.conv1 = DownConv(1, 64, drop_rate, batch_norm_momentum)
        self.mp1 = nn.MaxPool2d(2)

        self.conv2 = DownConv(64, 128, drop_rate, batch_norm_momentum)
        self.mp2 = nn.MaxPool2d(2)

        self.conv3 = DownConv(128, 256, drop_rate, batch_norm_momentum)
        self.mp3 = nn.MaxPool2d(2)

        # Bottom
        self.conv4 = DownConv(256, 256, drop_rate, batch_norm_momentum)

        # Upsampling path
        self.up1 = UpConv(512, 256, drop_rate, batch_norm_momentum)
        self.up2 = UpConv(384, 128, drop_rate, batch_norm_momentum)
        self.up3 = UpConv(192, 64, drop_rate, batch_norm_momentum)

        self.conv9 = nn.Conv2d(64, 5, kernel_size=1, padding=0)
        init.xavier_normal_(self.conv9.weight)
        init.constant_(self.conv9.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the sUNet model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x1 = self.conv1(x)
        x2 = self.mp1(x1)

        x3 = self.conv2(x2)
        x4 = self.mp2(x3)

        x5 = self.conv3(x4)
        x6 = self.mp3(x5)

        # Bottom
        x7 = self.conv4(x6)

        # Up-sampling
        x8 = self.up1(x7, x5)
        x9 = self.up2(x8, x3)
        x10 = self.up3(x9, x1)

        outputs = self.conv9(x10)

        return outputs


def get_model() -> nn.Module:
    """
    Get the sUNet model wrapped with DataParallel and moved to the appropriate device.

    Returns:
        nn.Module: sUNet model wrapped with DataParallel and moved to the appropriate device.
    """
    return DataParallel(sUNet()).to(DEVICE)
