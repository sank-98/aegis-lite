# Small CNN for CIFAR-10 classification
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """A small CNN for CIFAR-10: Conv->ReLU->Pool x2, then FC->ReLU->FC."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))   # 32x32 -> 16x16
        x = self.pool(self.relu(self.conv2(x)))   # 16x16 -> 8x8
        x = x.view(x.size(0), -1)                 # flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
