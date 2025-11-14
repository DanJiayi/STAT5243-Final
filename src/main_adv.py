import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from models import *
from utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def supervised_contrastive_loss(z, y, temperature=0.1):
    """
    z: (N, D) embeddings
    y: (N,) labels
    """
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


data = np.load("../data/frog_zeb_pca100.npz", allow_pickle=True)
X_s = data["X_s"]
y_s = data["y_s"]
X_t = data["X_t"]
y_t = data["y_t"]

X_s_tensor = torch.tensor(X_s, dtype=torch.float32)
y_s_tensor = torch.tensor(y_s, dtype=torch.long)
X_t_tensor = torch.tensor(X_t, dtype=torch.float32)
y_t_tensor = torch.tensor(y_t, dtype=torch.long)

src_loader = DataLoader(TensorDataset(X_s_tensor, y_s_tensor), batch_size=128, shuffle=True)
tgt_loader = DataLoader(TensorDataset(X_t_tensor, y_t_tensor), batch_size=128, shuffle=True)
tgt_test_loader = DataLoader(TensorDataset(X_t_tensor, y_t_tensor), batch_size=256, shuffle=False)

input_dim = X_s.shape[1]
num_classes = len(np.unique(y_s))


def train_source(Ms, C, loader, epochs=20, lr=1e-3, lambda_con=0.2):
    """
    Add source-only Supervised Contrastive Loss.
    """
    Ms.to(device); C.to(device)

    ce = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(list(Ms.parameters()) + list(C.parameters()), lr=lr)

    for ep in range(1, epochs+1):
        Ms.train(); C.train()

        total_ce = 0.0
        total_con = 0.0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            # forward
            z = Ms(xb)
            logits = C(z)

            # CE
            ce_loss = ce(logits, yb)

            # SupCon
            con_loss = supervised_contrastive_loss(z, yb)

            loss = ce_loss + lambda_con * con_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_ce += ce_loss.item()
            total_con += con_loss.item()

        print(f"[Stage A][Epoch {ep}] CE={total_ce:.4f}  Con={total_con:.4f}")

    return Ms, C

def train_adda(Ms, C, src_loader, tgt_loader, epochs=20, lr_disc=1e-4, lr_mt=2e-4):
    # Freeze Ms & C
    Ms.eval()
    for p in Ms.parameters(): p.requires_grad_(False)
    C.eval()
    for p in C.parameters(): p.requires_grad_(False)

    # Initialize Mt from Ms
    Mt = Encoder(input_dim, hidden_dim=64, latent_dim=64).to(device)
    Mt.load_state_dict(Ms.state_dict())

    # domain discriminator
    D = DomainClassifier(latent_dim=64, hidden_dim=128).to(device)

    bce = nn.BCEWithLogitsLoss()
    opt_D = torch.optim.Adam(D.parameters(), lr=lr_disc, betas=(0.5, 0.9))
    opt_Mt = torch.optim.Adam(Mt.parameters(), lr=lr_mt, betas=(0.5, 0.9))

    src_iter = iter(src_loader)

    for ep in range(1, epochs+1):
        Mt.train(); D.train()

        total_d = 0.0
        total_mt = 0.0

        for x_t, _ in tgt_loader:
            x_t = x_t.to(device)

            try:
                x_s, _ = next(src_iter)
            except StopIteration:
                src_iter = iter(src_loader)
                x_s, _ = next(src_iter)

            x_s = x_s.to(device)

            # features
            with torch.no_grad():
                f_s = Ms(x_s)      # frozen source encoder
            f_t = Mt(x_t)

            # Train D: source=1, target=0
            y_s = torch.ones(f_s.size(0), 1, device=device)
            y_t = torch.zeros(f_t.size(0), 1, device=device)

            log_s = D(f_s.detach())
            log_t = D(f_t.detach())
            loss_D = bce(log_s, y_s) + bce(log_t, y_t)

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # Train Mt: fool D (want D(f_t)=1)
            f_t = Mt(x_t)
            log_t2 = D(f_t)
            loss_Mt = bce(log_t2, torch.ones_like(y_t))

            opt_Mt.zero_grad()
            loss_Mt.backward()
            opt_Mt.step()

            total_d  += loss_D.item()
            total_mt += loss_Mt.item()

        print(f"[Stage B][Epoch {ep}] loss_D={total_d:.4f}  loss_Mt={total_mt:.4f}")

    return Mt


if __name__ == "__main__":
    Ms = Encoder(input_dim, hidden_dim=64, latent_dim=64)
    C  = LabelClassifier(latent_dim=64, num_classes=num_classes)

    print("\n===== Stage A: Train Ms + C (with SupCon) =====")
    Ms, C = train_source(Ms, C, src_loader, epochs=20, lr=1e-3, lambda_con=0.2)

    print("\n===== Stage B: ADDA Adaptation (train Mt + D) =====")
    Mt = train_adda(Ms, C, src_loader, tgt_loader, epochs=20)

    print("\n===== Final Evaluation (target only) =====")
    acc_final = 0
    total = 0
    Mt.eval(); C.eval()
    with torch.no_grad():
        for x_t, y_t in tgt_test_loader:
            x_t = x_t.to(device)
            y_t = y_t.to(device)
            logits = C(Mt(x_t))
            pred = logits.argmax(1)
            acc_final += (pred == y_t).sum().item()
            total += y_t.size(0)

    print("Final Target Accuracy =", acc_final / total)
