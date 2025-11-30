import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models import MLP

# =========================
# Load Data (same as yours)
# =========================
data = np.load("../data/frog_zeb_processed.npz", allow_pickle=True)
X_s = data["X_s"]
y_s = data["y_s"]
X_t = data["X_t"]
y_t = data["y_t"]

X_s_tensor = torch.tensor(X_s, dtype=torch.float32)
y_s_tensor = torch.tensor(y_s, dtype=torch.long)
X_t_tensor = torch.tensor(X_t, dtype=torch.float32)
y_t_tensor = torch.tensor(y_t, dtype=torch.long)

# DataLoader
train_loader = DataLoader(TensorDataset(X_s_tensor, y_s_tensor), 
                          batch_size=128, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_t_tensor, y_t_tensor),
                          batch_size=256, shuffle=False)

input_dim = X_s.shape[1]
num_classes = len(np.unique(y_s))

# =========================
# Repeat 20 Times
# =========================
RUNS = 20
train_accs = []
test_accs = []

for run in range(RUNS):
    print(f"\n========== RUN {run+1}/{RUNS} ==========")

    # ----- your hyperparams -----
    num_epochs = 20
    h = 128
    lr = 0.0005 

    # ----- build model -----
    model = MLP(input_dim, hidden=h, num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # =========================
    # Train
    # =========================
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

    # =========================
    # Final Accuracies
    # =========================
    model.eval()
    with torch.no_grad():
        pred_s = model(X_s_tensor).argmax(1)
        train_accuracy = (pred_s == y_s_tensor).float().mean().item()

        pred_t = model(X_t_tensor).argmax(1)
        test_accuracy = (pred_t == y_t_tensor).float().mean().item()

    print(f"\nFinal Source Acc = {train_accuracy:.4f}")
    print(f"Final Target Acc = {test_accuracy:.4f}")

    train_accs.append(train_accuracy)
    test_accs.append(test_accuracy)

# =========================
# Print MEAN & STD
# =========================
train_accs = np.array(train_accs)
test_accs = np.array(test_accs)

print("\n==================== RESULTS (20 runs) ====================")
print(f"Source Acc Mean = {train_accs.mean():.4f}, Std = {train_accs.std():.4f}")
print(f"Target Acc Mean = {test_accs.mean():.4f}, Std = {test_accs.std():.4f}")
print("===========================================================\n")
