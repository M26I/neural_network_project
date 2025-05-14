import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load dataset
iris = load_iris()
X = iris.data  # Features
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
y = iris.target.reshape(-1, 1)  # Labels as column vector

# One-hot encode the target 
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print("Data shapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)


def relu(x):
    return np.maximum(0, x)

def relu_derivative(output):
    return output > 0


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True)) 
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy(y_true, y_pred):
    # Clip values to avoid log(0)
    y_pred = np.clip(y_pred, 1e-12, 1. - 1e-12)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


input_size = 4
hidden_size = 6
output_size = 3
learning_rate = 0.05
epochs = 2000

# Random weights and biases
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
b2 = np.zeros((1, output_size))


for epoch in range(epochs):
    # FORWARD PASS
    z1 = X_train @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    y_pred = softmax(z2)

    # LOSS
    loss = cross_entropy(y_train, y_pred)

    # BACKPROPAGATION
    error_output = y_pred - y_train  # shape (n, 3)
    dW2 = a1.T @ error_output
    db2 = np.sum(error_output, axis=0, keepdims=True)

    error_hidden = error_output @ W2.T * relu_derivative(z1)
    dW1 = X_train.T @ error_hidden
    db1 = np.sum(error_hidden, axis=0, keepdims=True)

    # UPDATE WEIGHTS
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    # LOG
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

# FORWARD ON TEST
z1_test = X_test @ W1 + b1
a1_test = relu(z1_test)
z2_test = a1_test @ W2 + b2
y_pred_test = softmax(z2_test)

# Convert one-hot predictions to class index
predicted_classes = np.argmax(y_pred_test, axis=1)
true_classes = np.argmax(y_test, axis=1)

accuracy = np.mean(predicted_classes == true_classes)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
