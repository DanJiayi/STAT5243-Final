import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models import *
from torch.autograd import Function

def contrastive_loss(z, y, temperature=0.1):
    z = F.normalize(z, dim=1)
    N = z.size(0)
    sim = torch.matmul(z, z.T) / temperature
    mask = torch.eye(N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -9e15)
    y = y.contiguous().view(-1, 1)
    pos_mask = (y == y.T).float()
    exp_sim = torch.exp(sim)
    denom = exp_sim.sum(dim=1, keepdim=True)
    numerator = exp_sim * pos_mask
    loss = -torch.log((numerator.sum(dim=1) + 1e-9) / (denom.squeeze() + 1e-9))
    return loss.mean()


# ====================
# Load data (same)
# ====================
data = np.load("../data/frog_zeb_processed.npz", allow_pickle=True)
X_s = data["X_s"]
y_s = data["y_s"]
X_t = data["X_t"]
y_t = data["y_t"]

d_s = np.ones_like(y_s)
d_t = np.zeros_like(y_t)

X_all = np.concatenate([X_s, X_t], 0)
d_all = np.concatenate([d_s, d_t], 0)
y_all = np.concatenate([y_s, -np.ones_like(y_t)], 0)

X_all_tensor = torch.tensor(X_all, dtype=torch.float32)
y_all_tensor = torch.tensor(y_all, dtype=torch.long)
d_all_tensor = torch.tensor(d_all, dtype=torch.float32)

X_s_tensor = torch.tensor(X_s, dtype=torch.float32)
y_s_tensor = torch.tensor(y_s, dtype=torch.long)
X_t_tensor = torch.tensor(X_t, dtype=torch.float32)
y_t_tensor = torch.tensor(y_t, dtype=torch.long)

batch_size = 64
train_loader = DataLoader(
    TensorDataset(X_all_tensor, y_all_tensor, d_all_tensor),
    batch_size=batch_size,
    shuffle=True
)


# ====================
# GRL layer
# ====================
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# RUN 20 times
# ============================================================
RUNS = 20
results_tgt, results_src = [], []

print(f"\n================ RUNNING {RUNS} TIMES ================\n")

for run in range(RUNS):
    print(f"\n######## RUN {run+1}/{RUNS} ########\n")

    # --------------------------
    # Hyperparameters (same)
    # --------------------------
    input_dim = X_s.shape[1]
    num_classes = len(np.unique(y_s))
    latent_dim = 256
    hidden_dim = 512
    lambda_d = 0.05
    lambda_contrastive = 0.1
    use_contrastive = True
    num_epochs = 10
    lr = 0.00005

    # --------------------------
    # Build models
    # --------------------------
    encoder = Encoder(input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
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

    # =====================================================
    # Stage 1: Train GRL
    # =====================================================
    for epoch in range(num_epochs):
        encoder.train(); label_clf.train(); domain_clf.train()

        total_loss = total_y = total_d = total_con = 0

        for xb, yb, db in train_loader:
            xb, yb, db = xb.to(device), yb.to(device), db.to(device)
            z = encoder(xb)

            src_mask = (yb != -1)
            logits_y = label_clf(z)
            y_loss = criterion_y(logits_y[src_mask], yb[src_mask]) if src_mask.any() else torch.tensor(0.0, device=device)

            z_rev = grad_reverse(z)
            d_loss = criterion_d(domain_clf(z_rev).squeeze(1), db)

            if use_contrastive and src_mask.sum() > 1:
                con_loss = contrastive_loss(z[src_mask], yb[src_mask])
            else:
                con_loss = torch.tensor(0.0, device=device)

            loss = y_loss + lambda_d*d_loss + lambda_contrastive*con_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Final GRL acc
    encoder.eval(); label_clf.eval()
    with torch.no_grad():
        tgt_acc_before = (
            label_clf(encoder(X_t_tensor.to(device))).argmax(1).cpu() == y_t_tensor
        ).float().mean().item()

    # =====================================================
    # Stage 2: Freeze encoder + Weighted FT
    # =====================================================
    for p in encoder.parameters():
        p.requires_grad_(False)

    # compute weights
    domain_clf.eval()
    with torch.no_grad():
        z_s = encoder(X_s_tensor.to(device))
        p_source = torch.sigmoid(domain_clf(z_s)).cpu().numpy().flatten()

    eps = 1e-4
    w = 1/(p_source+eps)
    w /= w.mean()

    ft_loader = DataLoader(
        TensorDataset(X_s_tensor, y_s_tensor, torch.tensor(w, dtype=torch.float32)),
        batch_size=128, shuffle=True
    )

    for p in label_clf.parameters():
        p.requires_grad_(True)

    optimizer_ft = torch.optim.Adam(label_clf.parameters(), lr=1e-4)

    for epoch in range(25):
        for xb, yb, wb in ft_loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            with torch.no_grad():
                z = encoder(xb)
            logits = label_clf(z)
            ce = F.cross_entropy(logits, yb, reduction="none")
            loss = (ce * wb).mean()

            optimizer_ft.zero_grad()
            loss.backward()
            optimizer_ft.step()

    # =====================
    # Final evaluation
    # =====================
    encoder.eval(); label_clf.eval()
    with torch.no_grad():
        tgt_acc_after = (
            label_clf(encoder(X_t_tensor.to(device))).argmax(1).cpu() == y_t_tensor
        ).float().mean().item()

        src_acc_after = (
            label_clf(encoder(X_s_tensor.to(device))).argmax(1).cpu() == y_s_tensor
        ).float().mean().item()

    print(f"[RUN {run+1}] Final Target Acc = {tgt_acc_after:.4f}, Final Source Acc = {src_acc_after:.4f}")


    results_tgt.append(tgt_acc_after)
    results_src.append(src_acc_after)


# ============================================================
# Print final mean and std
# ============================================================
results_tgt = np.array(results_tgt)
results_src = np.array(results_src)
print("\n================ RESULTS ================")
print("Target Acc Mean =", results_tgt.mean())
print("Target Acc Std  =", results_tgt.std())
print("Source Acc Mean =", results_src.mean())
print("Source Acc Std  =", results_src.std())
print("=========================================\n")
