# extract_features_kaggle.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from gtzan_kaggle_dataset import GTZANKaggleImages, GENRES
from musicrecnet import MusicRecNet

def extract_dense2(root_dir, model_path="musicrecnet_best.pt"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = GTZANKaggleImages(root_dir)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    model = MusicRecNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_feats = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            _, feats = model(imgs)
            all_feats.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())

    X = np.concatenate(all_feats)
    y = np.concatenate(all_labels)

    np.save("dense2_features.npy", X)
    np.save("dense2_labels.npy", y)
    print("Saved Dense_2 features.")

if __name__ == "__main__":
    extract_dense2("Data/images_original")
