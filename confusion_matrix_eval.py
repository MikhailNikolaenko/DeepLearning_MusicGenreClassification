# confusion_matrix_eval.py
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from gtzan_kaggle_dataset import GTZANKaggleImages, GENRES
from musicrecnet import MusicRecNet
from torch.utils.data import DataLoader

ROOT = "Data/images_original"

dataset = GTZANKaggleImages(ROOT)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MusicRecNet().to(device)
model.load_state_dict(torch.load("musicrecnet_best.pt"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits, _ = model(imgs)
        preds = logits.argmax(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=GENRES, yticklabels=GENRES)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
