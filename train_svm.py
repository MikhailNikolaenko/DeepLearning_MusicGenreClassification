import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

def train_svm():
    X = np.load("dense2_features.npy")
    y = np.load("dense2_labels.npy")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = SVC(
        kernel='rbf',
        C=5,
        gamma='scale',
        probability=True   # VERY IMPORTANT for softmax-like output
    )

    clf.fit(X_train, y_train)

    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)

    print("SVM validation accuracy:", acc)
    joblib.dump(clf, "svm_dense2.joblib")
    print("Saved SVM model.")

if __name__ == "__main__":
    train_svm()
