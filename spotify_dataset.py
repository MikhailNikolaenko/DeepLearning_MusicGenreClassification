# spotify_dataset.py
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

GENRES_SPOTIFY = [
    "Classical",
    "Country",
    "EDM",
    "Hip Hop",
    "Jazz",
    "Latin",
    "Metal",
    "Pop",
    "RnB",
    "Rock"
]

class SpotifyImages(Dataset):
    def __init__(self, root_dir, image_size=128):
        self.root_dir = root_dir
        self.image_size = image_size

        # ───────────────────────────────────────────────
        # 1) Define transform AFTER cropping
        # ───────────────────────────────────────────────
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        self.paths = []
        self.labels = []

        # ───────────────────────────────────────────────
        # 2) Scan folders
        # ───────────────────────────────────────────────
        for label_idx, genre in enumerate(GENRES_SPOTIFY):
            gdir = os.path.join(root_dir, genre)
            if not os.path.exists(gdir):
                continue

            for fname in os.listdir(gdir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.paths.append(os.path.join(gdir, fname))
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path  = self.paths[idx]
        label = self.labels[idx]

        img = Image.open(path).convert("RGB")

        # ───────────────────────────────────────────────
        # 3) Crop out Spotify’s bad left/bottom edges
        # Original: 725 × 569  
        # Desired: 677 × 533  
        # That means:
        #   remove 48 px from LEFT  
        #   remove 36 px from BOTTOM
        # ───────────────────────────────────────────────
        w, h = img.size              # (725, 569)
        crop_left = 48
        crop_bottom = 36

        img = img.crop((
            crop_left,          # left
            0,                  # top
            w,                  # right
            h - crop_bottom     # bottom
        ))
        # Now ~677×533

        # apply transform
        img = self.transform(img)

        return img, label
