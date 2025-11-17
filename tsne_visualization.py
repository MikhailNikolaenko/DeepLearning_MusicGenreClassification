from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from gtzan_kaggle_dataset import GENRES

X = np.load("dense2_features.npy")
y = np.load("dense2_labels.npy")

X_emb = TSNE(n_components=2, perplexity=35, learning_rate=200).fit_transform(X)

plt.figure(figsize=(8,6))

for i, genre in enumerate(GENRES):
    plt.scatter(X_emb[y==i,0], X_emb[y==i,1], s=12, label=genre)

plt.legend(markerscale=2)
plt.title("t-SNE of Dense_2 Embeddings")
plt.tight_layout()
plt.savefig("tsne_dense2.png")
plt.show()
