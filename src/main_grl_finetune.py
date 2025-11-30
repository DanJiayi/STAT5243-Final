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

#data = np.load("../data/frog_zeb_pca100.npz", allow_pickle=True)
data = np.load("../data/frog_zeb_processed.npz", allow_pickle=True)
X_s = data["X_s"]
y_s = data["y_s"]
X_t = data["X_t"]
y_t = data["y_t"]

print("X_s:", X_s.shape, "y_s:", y_s.shape)
print("X_t:", X_t.shape, "y_t:", y_t.shape)

d_s = np.ones_like(y_s, dtype=np.int64)
d_t = np.zeros_like(y_t, dtype=np.int64)

X_all = np.concatenate([X_s, X_t], axis=0)
d_all = np.concatenate([d_s, d_t], axis=0)
y_t_dummy = -np.ones_like(y_t, dtype=np.int64)
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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = X_s.shape[1]
    num_classes = len(np.unique(y_s))
    latent_dim = 256#64
    hidden_dim = 512#64
    lambda_d = 0.05#0.1
    lambda_contrastive = 0.1#0.2
    use_contrastive = True
    num_epochs = 10 #20
    lr = 5e-5#1e-4

    ablation_mode = False
    if ablation_mode:
        lambda_d = 0.0
        use_contrastive = False

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

    # ===========================
    # Stage 1: Standard GRL training
    # ===========================
    for epoch in range(num_epochs):
        encoder.train()
        label_clf.train()
        domain_clf.train()

        total_loss = 0.0
        total_y_loss = 0.0
        total_d_loss = 0.0
        total_con_loss = 0.0

        for xb, yb, db in train_loader:
            xb, yb, db = xb.to(device), yb.to(device), db.to(device)

            z = encoder(xb)

            logits_y = label_clf(z)
            src_mask = (yb != -1)
            if src_mask.any():
                y_loss = criterion_y(logits_y[src_mask], yb[src_mask])
            else:
                y_loss = torch.tensor(0.0, device=device)

            z_rev = grad_reverse(z, lambd=1.0)
            logits_d = domain_clf(z_rev).squeeze(1)
            d_loss = criterion_d(logits_d, db)

            if use_contrastive and src_mask.sum() > 1:
                z_src = z[src_mask]
                y_src = yb[src_mask]
                con_loss = contrastive_loss(z_src, y_src)
            else:
                con_loss = torch.tensor(0.0, device=device)

            loss = y_loss + lambda_d * d_loss + lambda_contrastive * con_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_y_loss += y_loss.item()
            total_d_loss += d_loss.item()
            total_con_loss += con_loss.item()

        avg_loss = total_loss / len(train_loader)

        encoder.eval()
        label_clf.eval()
        with torch.no_grad():
            z_s = encoder(X_s_tensor.to(device))
            logits_s = label_clf(z_s)
            src_acc = (logits_s.argmax(1).cpu() == y_s_tensor).float().mean().item()

            z_t = encoder(X_t_tensor.to(device))
            logits_t = label_clf(z_t)
            tgt_acc = (logits_t.argmax(1).cpu() == y_t_tensor).float().mean().item()

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Loss={avg_loss:.4f} | y={total_y_loss/len(train_loader):.4f} | "
            f"d={total_d_loss/len(train_loader):.4f} | con={total_con_loss/len(train_loader):.4f} | "
            f"SrcAcc={src_acc:.4f} | TgtAcc={tgt_acc:.4f}"
        )

    # ===========================
    # Stage 1 Final Acc
    # ===========================
    encoder.eval(); label_clf.eval()
    with torch.no_grad():
        pred_s = label_clf(encoder(X_s_tensor.to(device))).argmax(1).cpu()
        pred_t = label_clf(encoder(X_t_tensor.to(device))).argmax(1).cpu()
        src_acc_before = (pred_s == y_s_tensor).float().mean().item()
        tgt_acc_before = (pred_t == y_t_tensor).float().mean().item()

    print("\n===== BEFORE Fine-Tuning =====")
    print(f"Source Acc = {src_acc_before:.4f}")
    print(f"Target Acc = {tgt_acc_before:.4f}")

    # ===========================
    # Stage 2: Freeze encoder, compute weights from domain discriminator
    # ===========================
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    domain_clf.eval()
    with torch.no_grad():
        z_s = encoder(X_s_tensor.to(device))
        p_source = torch.sigmoid(domain_clf(z_s)).cpu().numpy().flatten()

    eps = 1e-4
    w = 1.0 / (p_source + eps)
    w = w / w.mean()

    w_tensor = torch.tensor(w, dtype=torch.float32)

    ft_loader = DataLoader(
        TensorDataset(X_s_tensor, y_s_tensor, w_tensor),
        batch_size=128, shuffle=True
    )

    # ===========================
    # Stage 2: Fine-tune classifier only
    # ===========================
    for p in label_clf.parameters():
        p.requires_grad_(True)

    optimizer_ft = torch.optim.Adam(label_clf.parameters(), lr=1e-4)

    ft_epochs = 30
    print("\n===== Fine-Tuning Classifier (Weighted CE) =====")
    for epoch in range(ft_epochs):
        label_clf.train()
        total = 0

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

        # ===== 每个 epoch 后打印 ACC（NEW） =====
        label_clf.eval()
        with torch.no_grad():
            # source acc
            pred_s_ft = label_clf(encoder(X_s_tensor.to(device))).argmax(1).cpu()
            src_acc_ft = (pred_s_ft == y_s_tensor).float().mean().item()

            # target acc
            pred_t_ft = label_clf(encoder(X_t_tensor.to(device))).argmax(1).cpu()
            tgt_acc_ft = (pred_t_ft == y_t_tensor).float().mean().item()

        print(f"FT Epoch {epoch+1}/{ft_epochs} "
            f"| Loss={loss.item():.4f} "
            f"| SrcAcc={src_acc_ft:.4f} "
            f"| TgtAcc={tgt_acc_ft:.4f}")


    # ===========================
    # Final evaluation
    # ===========================
    encoder.eval()
    label_clf.eval()
    with torch.no_grad():
        pred_s_after = label_clf(encoder(X_s_tensor.to(device))).argmax(1).cpu()
        pred_t_after = label_clf(encoder(X_t_tensor.to(device))).argmax(1).cpu()

        src_acc_after = (pred_s_after == y_s_tensor).float().mean().item()
        tgt_acc_after = (pred_t_after == y_t_tensor).float().mean().item()

    print("\n===== AFTER Fine-Tuning =====")
    print(f"Source Acc = {src_acc_after:.4f}")
    print(f"Target Acc = {tgt_acc_after:.4f}")
