# gtzan_kaggle_dataset.py
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

class GTZANKaggleImages(Dataset):
    def __init__(self, root_dir, image_size=128):
        self.root_dir = root_dir
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        self.paths = []
        self.labels = []

        for label_idx, genre in enumerate(GENRES):
            gdir = os.path.join(root_dir, genre)
            for fname in os.listdir(gdir):
                if fname.endswith(".png") or fname.endswith(".jpg"):
                    self.paths.append(os.path.join(gdir, fname))
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]

        img = Image.open(path).convert("RGB")
        img = self.transform(img)

        return img, label
