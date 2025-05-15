import numpy as np
import matplotlib.pyplot as plt

# Load loss data
loss_manual = np.load("loss_manual.npy")
loss_pytorch = np.load("loss_pytorch.npy")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(loss_manual, label="Manual NN (NumPy)", color='blue')
plt.plot(loss_pytorch, label="PyTorch NN", color='orange')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
