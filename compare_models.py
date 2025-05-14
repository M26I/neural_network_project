import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load data
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# Normalize
X = (X - X.mean(axis=0)) / X.std(axis=0)

# One-hot encode
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
y_test_labels = np.argmax(y_test, axis=1)

# Load predictions
scratch_preds = np.load("scratch_predictions.npy")
pytorch_preds = np.load("pytorch_predictions.npy")

# Features to plot
feature_x = 0  # sepal length
feature_y = 2  # petal length

plt.figure(figsize=(18, 5))

# True Labels
plt.subplot(1, 3, 1)
plt.title("True Labels")
for label in range(3):
    idx = y_test_labels == label
    plt.scatter(X_test[idx, feature_x], X_test[idx, feature_y], label=f"Class {label}")
plt.xlabel(iris.feature_names[feature_x])
plt.ylabel(iris.feature_names[feature_y])
plt.legend()

# Scratch NN Predictions
plt.subplot(1, 3, 2)
plt.title("Scratch NN Predictions")
for label in range(3):
    idx = scratch_preds == label
    plt.scatter(X_test[idx, feature_x], X_test[idx, feature_y], label=f"Class {label}")
plt.xlabel(iris.feature_names[feature_x])
plt.ylabel(iris.feature_names[feature_y])
plt.legend()

# PyTorch Predictions
plt.subplot(1, 3, 3)
plt.title("PyTorch Predictions")
for label in range(3):
    idx = pytorch_preds == label
    plt.scatter(X_test[idx, feature_x], X_test[idx, feature_y], label=f"Class {label}")
plt.xlabel(iris.feature_names[feature_x])
plt.ylabel(iris.feature_names[feature_y])
plt.legend()

plt.tight_layout()
plt.show()
