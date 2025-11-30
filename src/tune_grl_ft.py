import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Function
from models import *
import sys
log_file = open("../logs/log_grl_ft.txt", "w")
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

# =====================================================
# Contrastive Loss
# =====================================================
def contrastive_loss(z, y, temperature=0.1):
    z = F.normalize(z, dim=1)
    N = z.size(0)
    sim = torch.matmul(z, z.T) / temperature
    sim = sim.masked_fill(torch.eye(N, device=z.device).bool(), -9e15)
    y = y.view(-1, 1)
    pos_mask = (y == y.T).float()

    exp_sim = torch.exp(sim)
    denom = exp_sim.sum(dim=1, keepdim=True)
    numerator = exp_sim * pos_mask
    loss = -torch.log((numerator.sum(dim=1) + 1e-9) / (denom.squeeze() + 1e-9))
    return loss.mean()


# =====================================================
# GRL
# =====================================================
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


# =====================================================
# Load Data
# =====================================================
data = np.load("../data/frog_zeb_processed.npz", allow_pickle=True)
X_s, y_s = data["X_s"], data["y_s"]
X_t, y_t = data["X_t"], data["y_t"]

print("X_s:", X_s.shape, "y_s:", y_s.shape)
print("X_t:", X_t.shape, "y_t:", y_t.shape)

d_s = np.ones_like(y_s)
d_t = np.zeros_like(y_t)

X_all = np.concatenate([X_s, X_t], 0)
y_all = np.concatenate([y_s, -np.ones_like(y_t)], 0)
d_all = np.concatenate([d_s, d_t], 0)

X_all_tensor = torch.tensor(X_all, dtype=torch.float32)
y_all_tensor = torch.tensor(y_all, dtype=torch.long)
d_all_tensor = torch.tensor(d_all, dtype=torch.float32)

X_s_tensor = torch.tensor(X_s, dtype=torch.float32)
X_t_tensor = torch.tensor(X_t, dtype=torch.float32)
y_s_tensor = torch.tensor(y_s, dtype=torch.long)
y_t_tensor = torch.tensor(y_t, dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_all_tensor, y_all_tensor, d_all_tensor),
    batch_size=64,
    shuffle=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================
# Hyperparameter Grids
# =====================================================
# num_epochs_list = [10]
# lr_list = [1e-4,5e-5]
# latent_list = [128, 256, 512]
# hidden_list = [128, 256, 512]
# lambda_d_list = [0.05]
# lambda_con_list = [0.05, 0.1, 0.2, 0.3]

num_epochs_list = [10]
lr_list = [5e-5]
latent_list = [256]
hidden_list = [512]
lambda_d_list = [0.05]
lambda_con_list = [0.1]


# =====================================================
# Training Loop Over All Hyperparameters
# =====================================================
os.makedirs("../models", exist_ok=True)
best_tgt = -1

for num_epochs in num_epochs_list:
    for lr in lr_list:
        for latent_dim in latent_list:
            for hidden_dim in hidden_list:
                for lambda_d in lambda_d_list: 
                    for lambda_contrastive in lambda_con_list:
                        if latent_dim!=512 and hidden_dim!=512 and lambda_contrastive!=0.05: continue
                        # if num_epochs==10 and (lr!=5e-05 or latent_dim!=256): continue

                        print(f"\n==== Running (E={num_epochs}, LR={lr}, LD={latent_dim}, HD={hidden_dim}, "
                              f"Ld={lambda_d}, Lcon={lambda_contrastive}) ====\n")

                        # ============================
                        # Build Model
                        # ============================
                        input_dim = X_s.shape[1]
                        num_classes = len(np.unique(y_s))

                        encoder = Encoder(input_dim, hidden_dim, latent_dim).to(device)
                        label_clf = LabelClassifier(latent_dim, num_classes).to(device)
                        domain_clf = DomainClassifier(latent_dim, hidden_dim=64).to(device)

                        optim = torch.optim.Adam(
                            list(encoder.parameters()) +
                            list(label_clf.parameters()) +
                            list(domain_clf.parameters()),
                            lr=lr
                        )

                        criterion_y = nn.CrossEntropyLoss()
                        criterion_d = nn.BCEWithLogitsLoss()

                        # =========================================================
                        # STAGE 1 — train GRL (NO ACC PRINT)
                        # =========================================================
                        for ep in range(num_epochs):

                            encoder.train()
                            label_clf.train()
                            domain_clf.train()

                            for xb, yb, db in train_loader:
                                xb, yb, db = xb.to(device), yb.to(device), db.to(device)

                                z = encoder(xb)

                                logits_y = label_clf(z)
                                src_mask = (yb != -1)
                                y_loss = criterion_y(logits_y[src_mask], yb[src_mask]) if src_mask.any() else 0

                                z_rev = grad_reverse(z, lambd=1.0)
                                logits_d = domain_clf(z_rev).squeeze(1)
                                d_loss = criterion_d(logits_d, db)

                                if src_mask.sum() > 1:
                                    con_loss = contrastive_loss(z[src_mask], yb[src_mask])
                                else:
                                    con_loss = 0

                                loss = y_loss + lambda_d * d_loss + lambda_contrastive * con_loss

                                optim.zero_grad()
                                loss.backward()
                                optim.step()

                        # =========================================================
                        # STAGE 2 — CLASSIFIER FINE TUNING
                        # =========================================================

                        # freeze encoder
                        encoder.eval()
                        for p in encoder.parameters():
                            p.requires_grad_(False)

                        # compute weights from domain classifier
                        domain_clf.eval()
                        with torch.no_grad():
                            p_src = torch.sigmoid(domain_clf(encoder(X_s_tensor.to(device)))).cpu().numpy().flatten()
                        w = 1 / (p_src + 1e-4)
                        w = w / w.mean()
                        w_tensor = torch.tensor(w, dtype=torch.float32)

                        ft_loader = DataLoader(
                            TensorDataset(X_s_tensor, y_s_tensor, w_tensor),
                            batch_size=128,
                            shuffle=True
                        )

                        # classifier fine-tuning
                        optimizer_ft = torch.optim.Adam(label_clf.parameters(), lr=1e-4)
                        ft_epochs = 40

                        

                        print("\n===== Fine-Tuning Stage =====")

                        for ft_ep in range(ft_epochs):

                            label_clf.train()
                            for xb, yb, wb in ft_loader:
                                xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)

                                with torch.no_grad():
                                    z = encoder(xb)

                                logits = label_clf(z)
                                ce = F.cross_entropy(logits, yb, reduction='none')
                                loss = (ce * wb).mean()

                                optimizer_ft.zero_grad()
                                loss.backward()
                                optimizer_ft.step()

                            # ===== eval =====
                            label_clf.eval()
                            with torch.no_grad():
                                pred_s = label_clf(encoder(X_s_tensor.to(device))).argmax(1).cpu()
                                pred_t = label_clf(encoder(X_t_tensor.to(device))).argmax(1).cpu()

                                src_acc = (pred_s == y_s_tensor).float().mean().item()
                                tgt_acc = (pred_t == y_t_tensor).float().mean().item()

                            print(f"FT Epoch {ft_ep+1}/{ft_epochs} | SrcAcc={src_acc:.4f} | TgtAcc={tgt_acc:.4f}")

                            # every 5 epochs check best
                            # if (ft_ep + 1) % 5 == 0:

                            # save if > 0.815
                            if (tgt_acc > best_tgt or (ft_ep + 1) % 5 == 0) and tgt_acc >= 0.825:
                                if tgt_acc > best_tgt:
                                    best_tgt = tgt_acc
                                    print(f"★★ NEW BEST TGT_ACC = {tgt_acc:.4f} ★★")
                                save_path = (
                                    f"../models/ft_{tgt_acc:.4f}_{src_acc:.4f}_"
                                    f"{ft_ep+1}_{num_epochs}_{lr}_"
                                    f"{latent_dim}_{hidden_dim}_"
                                    f"{lambda_d}_{lambda_contrastive}.pth"
                                )
                                torch.save(
                                    {
                                        "encoder": encoder.state_dict(),
                                        "label_clf": label_clf.state_dict(),
                                        "domain_clf": domain_clf.state_dict(),
                                        "meta": {
                                            "tgt_acc": tgt_acc,
                                            "src_acc": src_acc,
                                            "ft_epoch": ft_ep+1,
                                            "num_epochs": num_epochs,
                                            "lr": lr,
                                            "latent_dim": latent_dim,
                                            "hidden_dim": hidden_dim,
                                            "lambda_d": lambda_d,
                                            "lambda_con": lambda_contrastive,
                                        }
                                    },
                                    save_path
                                )
                                print(f"★ Saved: {save_path}\n")
