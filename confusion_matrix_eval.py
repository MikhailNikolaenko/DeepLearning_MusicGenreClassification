# confusion_matrix_eval.py

import torch
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from musicrecnet_lstm import MusicRecNetLSTM
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

lstm = MusicRecNetLSTM().to(device)
lstm.load_state_dict(torch.load("musicrecnet_lstm_best.pt", map_location=device))
lstm.eval()

svm = joblib.load("svm_dense2.joblib")   # SVM on extracted CNN features

# Ensemble weights
W_CNN  = 0.5
W_SVM  = 0.3
W_LSTM = 0.2

# -------------------------------------------------------------------
#  Collect predictions
# -------------------------------------------------------------------
all_labels = []
cnn_preds = []
svm_preds = []
lstm_preds = []
ens_preds = []

with torch.no_grad():

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        # CNN forward
        logits, feats = cnn(imgs)
        probs = torch.softmax(logits, dim=1)
        preds_cnn = probs.argmax(1)

        # LSTM
        logits_lstm, _ = lstm(imgs)
        probs_lstm = torch.softmax(logits_lstm, dim=1)
        preds_lstm = probs_lstm.argmax(1)

        # SVM forward (using CNN feature extractor)
        feats_np = feats.cpu().numpy()
        probs_svm = svm.predict_proba(feats_np)
        preds_svm = np.argmax(probs_svm, axis=1)

        # Ensemble
        probs_ens = (
            W_CNN  * probs.cpu().numpy() +
            W_SVM  * probs_svm +
            W_LSTM * probs_lstm.cpu().numpy()
        )
        preds_ens = np.argmax(probs_ens, axis=1)

        # Store
        all_labels.extend(labels.cpu().numpy())
        cnn_preds.extend(preds_cnn.cpu().numpy())
        svm_preds.extend(preds_svm)
        lstm_preds.extend(preds_lstm.cpu().numpy())
        ens_preds.extend(preds_ens)

# -------------------------------------------------------------------
#  Compute confusion matrices
# -------------------------------------------------------------------
cm_cnn = confusion_matrix(all_labels, cnn_preds)
cm_lstm = confusion_matrix(all_labels, lstm_preds)
cm_svm = confusion_matrix(all_labels, svm_preds)
cm_ens = confusion_matrix(all_labels, ens_preds)

# -------------------------------------------------------------------
#  Plot all 4
# -------------------------------------------------------------------
plt.figure(figsize=(28, 6))


# CNN
plt.subplot(1,4,1)
sns.heatmap(cm_cnn, annot=True, fmt="d", cmap="Blues",
            xticklabels=GENRES, yticklabels=GENRES)
plt.title("CNN")

plt.subplot(1,4,2)
sns.heatmap(cm_lstm, annot=True, fmt="d", cmap="Purples",
            xticklabels=GENRES, yticklabels=GENRES)
plt.title("LSTM")

plt.subplot(1,4,3)
sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Oranges",
            xticklabels=GENRES, yticklabels=GENRES)
plt.title("SVM")

plt.subplot(1,4,4)
sns.heatmap(cm_ens, annot=True, fmt="d", cmap="Greens",
            xticklabels=GENRES, yticklabels=GENRES)
plt.title("Ensemble (CNN + SVM + LSTM)")

plt.tight_layout()
plt.savefig("confusion_matrices_all_4.png", dpi=300)
plt.show()

print("\nSaved: confusion_matrices_all_4.png\n")