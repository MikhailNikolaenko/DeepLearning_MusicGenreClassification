# run_predictions_ensemble.py
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from math import pi
from musicrecnet import MusicRecNet
from musicrecnet_lstm import MusicRecNetLSTM
from torchvision import transforms
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load CNN
cnn = MusicRecNet().to(device)
cnn.load_state_dict(torch.load("musicrecnet_best.pt"))
cnn.eval()

# Load LSTM-CNN hybrid
lstm = MusicRecNetLSTM().to(device)
lstm.load_state_dict(torch.load("musicrecnet_lstm_best.pt"))
lstm.eval()

# Load SVM
svm = joblib.load("svm_dense2.joblib")

# ADD this:
GENRES = [
    "Classical","Country","EDM","Hip Hop","Jazz",
    "Latin","Metal","Pop","RnB","Rock"
]

# same transforms as training
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# -------------------------------------------------------
# VISUALIZATION FUNCTIONS
# -------------------------------------------------------

def plot_four_panel(cnn_probs, svm_probs, lstm_probs, ens_probs):
    plt.figure(figsize=(16,5))

    # CNN
    plt.subplot(1,4,1)
    plt.bar(GENRES, cnn_probs, color="royalblue")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0,1)
    plt.title("CNN Probabilities")

    # SVM
    plt.subplot(1,4,2)
    plt.bar(GENRES, svm_probs, color="darkorange")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0,1)
    plt.title("SVM Probabilities")

    # LSTM
    plt.subplot(1,4,3)
    plt.bar(GENRES, lstm_probs, color="red")
    plt.xticks(rotation=45, ha="right")
    plt.title("CNN + LSTM")

    # Ensemble
    plt.subplot(1,4,4)
    plt.bar(GENRES, ens_probs, color="seagreen")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0,1)
    plt.title("Ensemble Probabilities")

    plt.tight_layout()
    plt.savefig("four_panel_probs.png")
    plt.close()


def plot_heatmap(cnn_probs, svm_probs, lstm_probs, ens_probs):
    data = np.vstack([cnn_probs, svm_probs, lstm_probs, ens_probs])

    plt.figure(figsize=(8,4))
    plt.imshow(data, cmap="viridis", aspect="auto")

    plt.colorbar(label="Probability")

    plt.yticks([0,1,2,3], ["CNN", "SVM", "LSTM", "Ensemble"])
    plt.xticks(range(len(GENRES)), GENRES, rotation=45, ha="right")
    plt.title("Probability Heatmap")

    plt.tight_layout()
    plt.savefig("probability_heatmap.png")
    plt.close()


def plot_radar_chart(ens_probs):
    N = len(GENRES)

    # Close the loop
    values = np.concatenate([ens_probs, [ens_probs[0]]])
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2, color="crimson")
    ax.fill(angles, values, color="crimson", alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(GENRES, fontsize=8)

    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8"])

    plt.title("Ensemble Radar Chart", y=1.1)
    plt.tight_layout()
    plt.savefig("ensemble_radar.png")
    plt.close()


# -------------------------------------------------------
# PREDICTION FUNCTION
# -------------------------------------------------------

def predict(path, w_cnn=0.5, w_svm=0.3, w_lstm=0.2):
    img = Image.open(path).convert("RGB")

    # ------------------------------
    #  CROP to match training data
    # ------------------------------
    w, h = img.size
    crop_left = 48
    crop_bottom = 36

    img = img.crop((
        crop_left,
        0,
        w,
        h - crop_bottom
    ))

    # Now apply same training transforms
    x = transform(img).unsqueeze(0).to(device)

    # CNN
    with torch.no_grad():
        logits, feats = cnn(x)
        cnn_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # SVM
    svm_probs = svm.predict_proba(feats.cpu().numpy())[0]

        # LSTM
    with torch.no_grad():
        logits_lstm, _ = lstm(x)
        lstm_probs = torch.softmax(logits_lstm, dim=1).cpu().numpy()[0]

    # Ensemble
    ens_probs = w_cnn * cnn_probs + w_svm * svm_probs + w_lstm * lstm_probs

    # Print distributions
    print("\n===== CNN SOFTMAX =====")
    for g, p in zip(GENRES, cnn_probs):
        print(f"{g:10s}: {p*100:.2f}%")

    print("\n===== SVM (Platt Scaling) =====")
    for g, p in zip(GENRES, svm_probs):
        print(f"{g:10s}: {p*100:.2f}%")

    print("\n===== LSTM =====")
    for g,p in zip(GENRES, lstm_probs): print(f"{g:10s}: {p*100:.2f}%")

    print("\n===== ENSEMBLE (CNN + SVM + LSTM) =====")
    for g, p in zip(GENRES, ens_probs):
        print(f"{g:10s}: {p*100:.2f}%")

    # ---- VISUALIZATIONS ----
    plot_four_panel(cnn_probs, svm_probs, lstm_probs, ens_probs)
    plot_heatmap(cnn_probs, svm_probs, lstm_probs, ens_probs)
    plot_radar_chart(ens_probs)

    print("\nSaved visualizations:")
    print("  four_panel_probs.png")
    print("  probability_heatmap.png")
    print("  ensemble_radar.png\n")

    # Predictions
    print("Predictions:")
    print("  CNN:       ", GENRES[np.argmax(cnn_probs)])
    print("  SVM:       ", GENRES[np.argmax(svm_probs)])
    print("  LSTM:      ", GENRES[np.argmax(lstm_probs)])
    print("  Ensemble:  ", GENRES[np.argmax(ens_probs)])


if __name__ == "__main__":
    predict("Data/images_original/Pop/Pop1.png")