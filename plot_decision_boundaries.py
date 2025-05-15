import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from iris_nn import NeuralNetwork
import torch
from iris_nn_pytorch import IrisNet 

# Load and prepare data
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

X = (X - X.mean(axis=0)) / X.std(axis=0)
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
y_test_labels = np.argmax(y_test, axis=1)

# Reload models
nn = NeuralNetwork(input_size=4, hidden_size=6, output_size=3, learning_rate=0.05)
nn.train(X_train, y_train, epochs=2000)

torch_model = IrisNet()
torch_model.load_state_dict(torch.load("torch_model.pth"))
torch_model.eval()

# Select features
f1, f2 = 0, 2  # sepal length and petal length

# Mesh grid
x_min, x_max = X[:, f1].min() - 1, X[:, f1].max() + 1
y_min, y_max = X[:, f2].min() - 1, X[:, f2].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]

# Pad to 4D (your model expects 4 features)
X_grid = np.zeros((grid.shape[0], 4))
X_grid[:, f1] = grid[:, 0]
X_grid[:, f2] = grid[:, 1]

# Predictions from scratch model
pred_scratch = nn.predict(X_grid).reshape(xx.shape)

# Predictions from PyTorch model
with torch.no_grad():
    inputs = torch.tensor(X_grid, dtype=torch.float32)
    outputs = torch_model(inputs)
    pred_torch = torch.argmax(outputs, axis=1).numpy().reshape(xx.shape)

# Plot
plt.figure(figsize=(12, 5))

# Scratch
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, pred_scratch, cmap=plt.cm.Spectral, alpha=0.6)
plt.scatter(X_test[:, f1], X_test[:, f2], c=y_test_labels, edgecolor='k')
plt.title("Decision Boundary - Scratch NN")
plt.xlabel(iris.feature_names[f1])
plt.ylabel(iris.feature_names[f2])

# PyTorch
plt.subplot(1, 2, 2)
plt.contourf(xx, yy, pred_torch, cmap=plt.cm.Spectral, alpha=0.6)
plt.scatter(X_test[:, f1], X_test[:, f2], c=y_test_labels, edgecolor='k')
plt.title("Decision Boundary - PyTorch")
plt.xlabel(iris.feature_names[f1])
plt.ylabel(iris.feature_names[f2])

plt.tight_layout()
plt.show()
