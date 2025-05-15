import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

# Load and preprocess data
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
y_test_labels = np.argmax(y_test, axis=1)

class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, 6)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(6, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

model = IrisNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

y_train_labels = torch.tensor(np.argmax(y_train, axis=1), dtype=torch.long)

# Track loss
loss_history = []

# Train loop
for epoch in range(1, 2001):
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item()) 

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Evaluate
with torch.no_grad():
    predictions = model(X_test_tensor)
    predicted_labels = torch.argmax(predictions, dim=1).numpy()
    accuracy = np.mean(predicted_labels == y_test_labels)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


# Save the model
torch.save(model.state_dict(), "torch_model.pth")

# Save predictions
np.save("pytorch_predictions.npy", predicted_labels)

# Save loss history
np.save("loss_pytorch.npy", loss_history)
