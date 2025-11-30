import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models import *
from torch.autograd import Function

# ================================
# Contrastive loss
# ================================
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
    loss = -torch.log((numerator.sum(dim=1)+1e-9) / (denom.squeeze()+1e-9))
    return loss.mean()

# ================================
# Load data
# ================================
data = np.load("../data/frog_zeb_processed.npz", allow_pickle=True)
X_s = data["X_s"]
y_s = data["y_s"]
X_t = data["X_t"]
y_t = data["y_t"]

d_s = np.ones_like(y_s, dtype=np.int64)
d_t = np.zeros_like(y_t, dtype=np.int64)

X_all = np.concatenate([X_s, X_t], axis=0)
d_all = np.concatenate([d_s, d_t], axis=0)
y_t_dummy = -np.ones_like(y_t)
y_all = np.concatenate([y_s, y_t_dummy], axis=0)

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

# ================================
# GRL
# ================================
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

# ================================
# Single run function
# return: (src_acc_final, tgt_acc_final)
# ================================
def run_single_training(use_contrastive):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = X_s.shape[1]
    num_classes = len(np.unique(y_s))
    latent_dim = 64
    hidden_dim = 128
    lambda_d = 0.1
    lambda_contrastive = 0.25
    num_epochs = 15
    lr = 0.00005

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

    # ===================== Train =====================
    for epoch in range(num_epochs):
        encoder.train()
        label_clf.train()
        domain_clf.train()

        for xb, yb, db in train_loader:
            xb, yb, db = xb.to(device), yb.to(device), db.to(device)

            z = encoder(xb)
            logits_y = label_clf(z)
            src_mask = (yb != -1)

            # y loss
            if src_mask.any():
                y_loss = criterion_y(logits_y[src_mask], yb[src_mask])
            else:
                y_loss = torch.tensor(0.0, device=device)

            # domain loss
            z_rev = grad_reverse(z, lambd=1.0)
            d_loss = criterion_d(domain_clf(z_rev).squeeze(1), db)

            # contrastive loss
            if use_contrastive and src_mask.sum() > 1:
                con_loss = contrastive_loss(z[src_mask], yb[src_mask])
            else:
                con_loss = torch.tensor(0.0, device=device)

            loss = y_loss + lambda_d * d_loss + lambda_contrastive * con_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # ===================== Evaluation =====================
    encoder.eval()
    label_clf.eval()
    with torch.no_grad():
        # source
        z_s = encoder(X_s_tensor.to(device))
        logits_s = label_clf(z_s)
        src_acc_final = (logits_s.argmax(1).cpu() == y_s_tensor).float().mean().item()

        # target
        z_t = encoder(X_t_tensor.to(device))
        logits_t = label_clf(z_t)
        tgt_acc_final = (logits_t.argmax(1).cpu() == y_t_tensor).float().mean().item()

    return src_acc_final, tgt_acc_final


# ================================
# Run N × with contrastive
# ================================
N1 = 20
src_con, tgt_con = [], []

for i in range(N1):
    s, t = run_single_training(use_contrastive=True)
    src_con.append(s)
    tgt_con.append(t)
    print(f"[Contrastive] Run {i+1}/{N1}: Src={s:.4f}  Tgt={t:.4f}")

print("\n======= WITH CONTRASTIVE =======")
print(f"Source Mean = {np.mean(src_con):.4f}, Std = {np.std(src_con):.4f}")
print(f"Target Mean = {np.mean(tgt_con):.4f}, Std = {np.std(tgt_con):.4f}")


# ================================
# Run N × without contrastive
# ================================
N2 = 20
src_no, tgt_no = [], []

for i in range(N2):
    s, t = run_single_training(use_contrastive=False)
    src_no.append(s)
    tgt_no.append(t)
    print(f"[NO Contrastive] Run {i+1}/{N2}: Src={s:.4f}  Tgt={t:.4f}")

print("\n======= WITHOUT CONTRASTIVE =======")
print(f"Source Mean = {np.mean(src_no):.4f}, Std = {np.std(src_no):.4f}")
print(f"Target Mean = {np.mean(tgt_no):.4f}, Std = {np.std(tgt_no):.4f}")
