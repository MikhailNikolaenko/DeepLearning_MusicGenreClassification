import torch
import torch.nn as nn
import torch.nn.functional as F

class MusicRecNetLSTM(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # 3×3 kernels, filters: 32 → 64 → 128
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.25)

        # LSTM( input=128, hidden=200, layers=4 )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=200,
            num_layers=4,
            batch_first=False
        )

        # 4 layers × 200 hidden = 800 features
        self.flat_dim = 4 * 200

        # Dense layers
        self.fc1 = nn.Linear(self.flat_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x):
        # Conv1
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop(x)

        # Conv2
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop(x)

        # Conv3
        x = self.pool(F.relu(self.conv3(x)))
        x = self.drop(x)

        # CNN output: (B, 128, H, W)
        # LSTM expects (seq, batch, feature)

        # Flatten spatial dims into sequence:
        # (B, C, H, W) → (B, C, H*W)
        x = torch.flatten(x, 2)

        # Permute to (seq_len, batch, features=C)
        x = x.permute(2, 0, 1)

        # Run LSTM
        out, (hn, cn) = self.lstm(x)

        # hn shape: (num_layers, batch, hidden)
        # Flatten all layers:
        hn = hn.permute(1, 0, 2)    # (batch, layers, dim)
        hn = hn.reshape(hn.size(0), -1)

        # Dense
        x = F.relu(self.fc1(hn))
        features = self.fc2(x)
        logits = self.fc_out(features)

        return logits, features
