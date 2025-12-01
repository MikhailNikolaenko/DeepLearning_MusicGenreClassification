# musicrecnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MusicRecNet(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()

        # 3×3 kernels, filters: 32 → 64 → 128
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # dropout rate not provided → match typical value
        self.drop = nn.Dropout(0.25)

        # LSTM layer
        self.lstm1 = nn.LSTM(128, 200, 4)


        # four lstm layers, 20 * 4 = 80 
        # According to PyTorch, LSTM outputs the original output, h_n, c_n.
        # h_n represent final state and is of the shape (layers * dim, out features)
        self.flat_dim = 800
        

        # Dense (paper)
        self.fc1 = nn.Linear(self.flat_dim, 256)   # reasonable hidden size
        self.fc2 = nn.Linear(256, 128)             # Dense_2 (paper: extractor)
        self.fc_out = nn.Linear(128, num_classes)  # output logits

    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))
        x=  self.pool(x)
        x = self.drop(x)

        # Conv block 2
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(x)
        x = self.drop(x)

        # Conv block 3
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(x)
        x = self.drop(x)

        # LSTM First Block
        x = torch.flatten(x, 2, 3)
        x = x.permute(2, 0, 1)
        
        # LSTM takes 2D, or 3D inputs while CNN gives 4D output
        # So input dim needs to be changed to fit the LSTM.
        # According to PyTorch, Input has the following shape (Sequence Length, Batch Size, Input_size)
        # Since CNN output is (Batch, Channels, Width, Height), 
        out, (hn, cn) = self.lstm1(x)    


        # Flatten and set correct dimensions
        hn = hn.permute(0, 2, 1)
        hn = torch.flatten(hn, 0, 1)
        hn = hn.permute(1, 0)
        out = out.permute(1, 0, 2)
        out = torch.flatten(out, 1, 2)

        # Dense
        x = F.relu(self.fc1(hn))
        features = self.fc2(x)   # 128-dim representation (used later)
        logits = self.fc_out(features)

        # NOTE: No softmax, CrossEntropyLoss applies it.
        return logits, features
