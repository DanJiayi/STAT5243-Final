import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models import MLP

data = np.load("data/frog_zeb_data_processed.npz", allow_pickle=True)
X_s = data["X_s"]
y_s = data["y_s"]
X_t = data["X_t"]
y_t = data["y_t"]
print(X_s.min(),X_s.max(),X_t.min(),X_t.max())
print("X_s:", X_s.shape, "y_s:", y_s.shape)
print("X_t:", X_t.shape, "y_t:", y_t.shape)

X_s_tensor = torch.tensor(X_s, dtype=torch.float32)
y_s_tensor = torch.tensor(y_s, dtype=torch.long)
X_t_tensor = torch.tensor(X_t, dtype=torch.float32)
y_t_tensor = torch.tensor(y_t, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_s_tensor, y_s_tensor), 
                          batch_size=128, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_t_tensor, y_t_tensor),
                          batch_size=256, shuffle=False)

input_dim = X_s.shape[1]
num_classes = len(np.unique(y_s))
num_epochs = 20
h = 256
lr = 1e-3

model = MLP(input_dim, hidden=h, num_classes=num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for xb, yb in train_loader:
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)

    train_acc = correct / total
    print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | "
          f"Train Acc: {train_acc:.4f}")

model.eval()
with torch.no_grad():
    logits = model(X_s_tensor)
    y_pred_s = logits.argmax(1)
    train_accuracy = (y_pred_s == y_s_tensor).float().mean().item()

print("\n=== Final Training Accuracy (X_s) ===")
print(f"Train Acc: {train_accuracy:.4f}")
with torch.no_grad():
    logits = model(X_t_tensor)
    y_pred_t = logits.argmax(1)
    test_accuracy = (y_pred_t == y_t_tensor).float().mean().item()

print("\n=== Test Accuracy (X_t → y_t) ===")
print(f"Test Acc: {test_accuracy:.4f}")
