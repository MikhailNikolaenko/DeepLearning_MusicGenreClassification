# confusion_matrix_eval.py

import torch
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from spotify_dataset import SpotifyImages
from musicrecnet import MusicRecNet

# -------------------------------------------------------------------
#  GENRES (MUST match training order)
# -------------------------------------------------------------------
GENRES = [
    "Classical","Country","EDM","Hip Hop","Jazz",
    "Latin","Metal","Pop","RnB","Rock"
]

# -------------------------------------------------------------------
#  Load dataset
# -------------------------------------------------------------------
ROOT = "Data/images_original"

dataset = SpotifyImages(ROOT)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------------------
#  Load models
# -------------------------------------------------------------------
cnn = MusicRecNet().to(device)
cnn.load_state_dict(torch.load("musicrecnet_best.pt", map_location=device))
cnn.eval()

svm = joblib.load("svm_dense2.joblib")   # SVM on extracted CNN features

# Ensemble weights
W_CNN = 0.6
W_SVM = 0.4

# -------------------------------------------------------------------
#  Collect predictions
# -------------------------------------------------------------------
all_labels = []
cnn_preds = []
svm_preds = []
ens_preds = []

with torch.no_grad():

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        # CNN forward
        logits, feats = cnn(imgs)
        probs = torch.softmax(logits, dim=1)
        preds_cnn = probs.argmax(1)

        # SVM forward (using CNN feature extractor)
        feats_np = feats.cpu().numpy()
        probs_svm = svm.predict_proba(feats_np)
        preds_svm = np.argmax(probs_svm, axis=1)

        # Ensemble prediction
        probs_cnn_np = probs.cpu().numpy()
        probs_ens = W_CNN * probs_cnn_np + W_SVM * probs_svm
        preds_ens = np.argmax(probs_ens, axis=1)

        # Store
        all_labels.extend(labels.cpu().numpy())
        cnn_preds.extend(preds_cnn.cpu().numpy())
        svm_preds.extend(preds_svm)
        ens_preds.extend(preds_ens)

# -------------------------------------------------------------------
#  Compute confusion matrices
# -------------------------------------------------------------------
cm_cnn = confusion_matrix(all_labels, cnn_preds)
cm_svm = confusion_matrix(all_labels, svm_preds)
cm_ens = confusion_matrix(all_labels, ens_preds)

# -------------------------------------------------------------------
#  Plot all 3
# -------------------------------------------------------------------
plt.figure(figsize=(20, 6))

# CNN
plt.subplot(1, 3, 1)
sns.heatmap(cm_cnn, annot=True, fmt="d", cmap="Blues",
            xticklabels=GENRES, yticklabels=GENRES)
plt.title("CNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

# SVM
plt.subplot(1, 3, 2)
sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Oranges",
            xticklabels=GENRES, yticklabels=GENRES)
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

# Ensemble
plt.subplot(1, 3, 3)
sns.heatmap(cm_ens, annot=True, fmt="d", cmap="Greens",
            xticklabels=GENRES, yticklabels=GENRES)
plt.title("Ensemble Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.tight_layout()
plt.savefig("confusion_matrices_all.png", dpi=300)
plt.show()

print("\nSaved: confusion_matrices_all.png\n")
