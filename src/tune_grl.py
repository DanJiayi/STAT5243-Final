import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import os
from models import *
from utils import *

import sys
log_file = open("../logs/log_grl.txt", "w")
class Logger(object):
    def __init__(self, terminal, log):
        self.terminal = terminal
        self.log = log
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(sys.stdout, log_file)
sys.stderr = Logger(sys.stderr, log_file)


# ============================
# Contrastive loss
# ============================
def contrastive_loss(z, y, temperature=0.1):
    z = F.normalize(z, dim=1)
    N = z.size(0)
    sim = torch.matmul(z, z.T) / temperature
    mask = torch.eye(N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -9e15)
    y = y.view(-1, 1)
    pos_mask = (y == y.T).float()

    exp_sim = torch.exp(sim)
    denom = exp_sim.sum(dim=1, keepdim=True)
    numerator = exp_sim * pos_mask

    loss = -torch.log((numerator.sum(dim=1) + 1e-9)/(denom.squeeze() + 1e-9))
    return loss.mean()


# ============================
# Load data
# ============================
data = np.load("../data/frog_zeb_processed.npz", allow_pickle=True)
X_s = data["X_s"]; y_s = data["y_s"]
X_t = data["X_t"]; y_t = data["y_t"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_s_tensor = torch.tensor(X_s, dtype=torch.float32)
y_s_tensor = torch.tensor(y_s, dtype=torch.long)
X_t_tensor = torch.tensor(X_t, dtype=torch.float32)
y_t_tensor = torch.tensor(y_t, dtype=torch.long)

# mixed dataloader
batch_size = 64
d_s = np.ones_like(y_s)
d_t = np.zeros_like(y_t)
X_all = np.concatenate([X_s, X_t])
d_all = np.concatenate([d_s, d_t])
y_all = np.concatenate([y_s, -np.ones_like(y_t)])

train_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_all, dtype=torch.float32),
        torch.tensor(y_all, dtype=torch.long),
        torch.tensor(d_all, dtype=torch.float32)
    ), batch_size=batch_size, shuffle=True
)

# ============================
# GRL
# ============================
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.clone()
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


# ============================================================
# Hyperparameter search settings
# ============================================================
latent_list = [128]
hidden_list = [64, 128]
lambda_d_list = [0.05, 0.1, 0.2, 0.3]
lambda_con_list = [0.1, 0.15, 0.2, 0.25, 0.3]
lr_list = [5e-5, 1e-4, 1e-3]

num_epochs = 40
check_interval = 5  # only compare + save every 5 epochs

input_dim = X_s.shape[1]
num_classes = len(np.unique(y_s))

best_acc = -1
best_params = None

os.makedirs("../models", exist_ok=True)


# ============================================================
# Grid Search
# ============================================================
for latent_dim in latent_list:
    for hidden_dim in hidden_list:
        for lambda_d in lambda_d_list:
            for lambda_con in lambda_con_list:
                for lr in lr_list:

                    print(f"\n===== RUN: latent={latent_dim}, hidden={hidden_dim}, "
                          f"lambda_d={lambda_d}, lambda_con={lambda_con}, lr={lr} =====\n")

                    encoder = Encoder(input_dim, hidden_dim, latent_dim).to(device)
                    label_clf = LabelClassifier(latent_dim, num_classes).to(device)
                    domain_clf = DomainClassifier(latent_dim, hidden_dim=64).to(device)

                    optimizer = torch.optim.Adam(
                        list(encoder.parameters()) +
                        list(label_clf.parameters()) +
                        list(domain_clf.parameters()),
                        lr=lr
                    )

                    criterion_y = nn.CrossEntropyLoss()
                    criterion_d = nn.BCEWithLogitsLoss()

                    # ============================
                    # 30 Epoch Training
                    # ============================
                    for epoch in range(1, num_epochs+1):
                        encoder.train()
                        label_clf.train()
                        domain_clf.train()

                        total_loss = total_y = total_d = total_con = 0

                        for xb, yb, db in train_loader:
                            xb, yb, db = xb.to(device), yb.to(device), db.to(device)

                            z = encoder(xb)

                            # classification (source only)
                            src_mask = (yb != -1)
                            if src_mask.any():
                                logits_y = label_clf(z[src_mask])
                                y_loss = criterion_y(logits_y, yb[src_mask])
                            else:
                                y_loss = torch.tensor(0.0, device=device)

                            # domain loss
                            z_rev = grad_reverse(z, 1.0)
                            logits_d = domain_clf(z_rev).squeeze(1)
                            d_loss = criterion_d(logits_d, db)

                            # supervised contrastive loss
                            if src_mask.sum() > 1:
                                con_loss = contrastive_loss(z[src_mask], yb[src_mask])
                            else:
                                con_loss = torch.tensor(0.0, device=device)

                            loss = y_loss + lambda_d*d_loss + lambda_con*con_loss

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            total_loss += loss.item()
                            total_y += y_loss.item()
                            total_d += d_loss.item()
                            total_con += con_loss.item()

                        # ============================
                        # Compute acc every epoch
                        # ============================
                        encoder.eval(); label_clf.eval()
                        with torch.no_grad():
                            pred_s = label_clf(encoder(X_s_tensor.to(device))).argmax(1).cpu()
                            pred_t = label_clf(encoder(X_t_tensor.to(device))).argmax(1).cpu()
                            src_acc = (pred_s == y_s_tensor).float().mean().item()
                            tgt_acc = (pred_t == y_t_tensor).float().mean().item()

                        # print every epoch
                        print(f"Epoch {epoch}/{num_epochs} | "
                              f"Loss={total_loss/len(train_loader):.4f} "
                              f"| y={total_y/len(train_loader):.4f} "
                              f"| d={total_d/len(train_loader):.4f} "
                              f"| con={total_con/len(train_loader):.4f} "
                              f"| SrcAcc={src_acc:.4f} | TgtAcc={tgt_acc:.4f}")

                        # ============================
                        # Compare with best only every 5 epochs
                        # ============================
                        if epoch % check_interval == 0:
                            if tgt_acc > best_acc:
                                best_acc = tgt_acc
                                best_params = (latent_dim, hidden_dim,
                                               lambda_d, lambda_con, lr)
                                print(f"*** NEW BEST: {tgt_acc:.4f}_{src_acc:.4f}_{epoch}_{lr}_{latent_dim}_{hidden_dim}_{lambda_d}_{lambda_con} ***")
                                
                            if tgt_acc>0.81:
                                save_path = (
                                    f"../models/{tgt_acc:.4f}_{src_acc:.4f}_{epoch}_{lr}_"
                                    f"{latent_dim}_{hidden_dim}_{lambda_d}_{lambda_con}.pth"
                                )

                                torch.save(
                                    {
                                        "encoder": encoder.state_dict(),
                                        "label_clf": label_clf.state_dict(),
                                        "domain_clf": domain_clf.state_dict(),
                                        "params": best_params
                                    },
                                    save_path
                                )

                                print(f"*** NEW SAVED: {save_path} ***")

print("\n===== SEARCH DONE =====")
print("BEST Target Acc =", best_acc)
print("BEST Params =", best_params)
