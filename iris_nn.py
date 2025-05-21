# © 2025 M26I - For educational/portfolio use only
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.05):
        # He initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate
        self.loss_history = []

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return x > 0

    def softmax(self, x):
        exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-8)) / m

    def backward(self, X, y_true):
        m = y_true.shape[0]
        error_output = self.a2 - y_true
        dW2 = self.a1.T @ error_output / m
        db2 = np.sum(error_output, axis=0, keepdims=True) / m

        error_hidden = error_output @ self.W2.T * self.relu_derivative(self.a1)
        dW1 = X.T @ error_hidden / m
        db1 = np.sum(error_hidden, axis=0, keepdims=True) / m

        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        


    def train(self, X, y, epochs=1000):
     for epoch in range(1, epochs + 1):
        y_pred = self.forward(X)
        loss = self.compute_loss(y_pred, y)
        self.loss_history.append(loss)  
        self.backward(X, y)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")


    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def evaluate(self, X, y_true_labels):
        y_pred_labels = self.predict(X)
        acc = np.mean(y_pred_labels == y_true_labels)
        print(f"Test Accuracy: {acc * 100:.2f}%")


# Load and preprocess data
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# Normalize
X = (X - X.mean(axis=0)) / X.std(axis=0)

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)




# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
y_test_labels = np.argmax(y_test, axis=1)

# Train
nn = NeuralNetwork(input_size=4, hidden_size=6, output_size=3, learning_rate=0.05)
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
nn.train(X_train, y_train, epochs=2000)





# Evaluate
nn.evaluate(X_test, y_test_labels)

# Predict on test data
y_pred_labels = nn.predict(X_test)

# Save predictions for comparison
np.save("scratch_predictions.npy", y_pred_labels)

# Save manual model loss curve
np.save("loss_manual.npy", nn.loss_history)


# Get activations for test inputs
activations = nn.relu(X_test @ nn.W1 + nn.b1)

# Visualize activations for first 5 test samples
plt.figure(figsize=(12, 5))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.bar(range(activations.shape[1]), activations[i])
    plt.title(f"Sample {i+1}\nLabel: {y_test_labels[i]}")
    plt.xticks(range(activations.shape[1]), [f"H{i+1}" for i in range(activations.shape[1])])
    plt.ylim(0, np.max(activations))  # consistent y-axis
plt.suptitle("Hidden Layer Activations (Test Samples)")
plt.tight_layout()
plt.show()


# features to plot
feature_x = 0  
feature_y = 2  

# Predict on test data
y_pred_labels = nn.predict(X_test)

# Plot True Labels
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("True Labels")
for label in range(3):
    idx = y_test_labels == label
    plt.scatter(X_test[idx, feature_x], X_test[idx, feature_y], label=f"Class {label}")
plt.xlabel(iris.feature_names[feature_x])
plt.ylabel(iris.feature_names[feature_y])
plt.legend()

# Plot Predicted Labels
plt.subplot(1, 2, 2)
plt.title("Predicted Labels")
for label in range(3):
    idx = y_pred_labels == label
    plt.scatter(X_test[idx, feature_x], X_test[idx, feature_y], label=f"Class {label}")
plt.xlabel(iris.feature_names[feature_x])
plt.ylabel(iris.feature_names[feature_y])
plt.legend()

plt.tight_layout()
plt.show()


# Visualize weights of input → hidden layer
plt.figure(figsize=(10, 4))
for i in range(nn.W1.shape[1]):  # for each hidden neuron
    plt.subplot(1, nn.W1.shape[1], i + 1)
    plt.bar(range(nn.W1.shape[0]), nn.W1[:, i])
    plt.xticks(range(nn.W1.shape[0]), iris.feature_names, rotation=45)
    plt.title(f"Neuron {i+1}")
    plt.tight_layout()
plt.suptitle("Input → Hidden Layer Weights")
plt.show()

