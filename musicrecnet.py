# musicrecnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MusicRecNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # 3×3 kernels, filters: 32 → 64 → 128
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # dropout rate not provided → match typical value
        self.drop = nn.Dropout(0.25)

        # flatten dimension: 128 * 16 * 16 = 32768
        self.flat_dim = 128 * 16 * 16

        # Dense (paper)
        self.fc1 = nn.Linear(self.flat_dim, 256)   # reasonable hidden size
        self.fc2 = nn.Linear(256, 128)             # Dense_2 (paper: extractor)
        self.fc_out = nn.Linear(128, num_classes)  # output logits

    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop(x)

        # Conv block 2
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop(x)

        # Conv block 3
        x = self.pool(F.relu(self.conv3(x)))
        x = self.drop(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Dense
        x = F.relu(self.fc1(x))

        features = self.fc2(x)   # 128-dim representation (used later)
        logits = self.fc_out(features)

        # NOTE: No softmax, CrossEntropyLoss applies it.
        return logits, features
