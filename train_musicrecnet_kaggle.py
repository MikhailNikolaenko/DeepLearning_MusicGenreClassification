# train_musicrecnet_kaggle.py
import torch
from torch.utils.data import DataLoader, random_split
from gtzan_kaggle_dataset import GTZANKaggleImages
from musicrecnet import MusicRecNet
import matplotlib.pyplot as plt

train_acc_hist = []
val_acc_hist = []

def train_musicrecnet(root_dir, batch_size=64, epochs=50, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = GTZANKaggleImages(root_dir)

    total = len(dataset)
    train_len = int(0.8 * total)
    val_len = total - train_len

    train_set, val_set = random_split(
        dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = MusicRecNet().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0)

    best_val_acc = 0

    for epoch in range(epochs):
        model.train()
        total_correct = 0
        total_samples = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        train_acc = total_correct / total_samples

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits, _ = model(imgs)
                preds = logits.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "musicrecnet_best.pt")

        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

    print(f"Best validation accuracy: {best_val_acc:.4f}")

    return model, val_loader, device


def plot_curves(train_acc, val_acc):
    plt.figure(figsize=(8,5))
    plt.plot(train_acc, label="Train Acc")
    plt.plot(val_acc, label="Validation Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("MusicRecNet Training Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_curve.png")
    plt.show()


if __name__ == "__main__":
    ROOT = "Data/images_original"

    model, val_loader, device = train_musicrecnet(ROOT)
    plot_curves(train_acc_hist, val_acc_hist)
