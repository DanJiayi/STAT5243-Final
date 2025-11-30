import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from models import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


# ========================= LOAD DATA =========================
data = np.load("../data/frog_zeb_processed.npz", allow_pickle=True)
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


# ========================= TRAIN SOURCE =========================
def train_source(Ms, C, loader, epochs=20, lr=1e-3, lambda_con=0.2):
    Ms.to(device); C.to(device)
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(list(Ms.parameters()) + list(C.parameters()), lr=lr)

    for ep in range(1, epochs+1):
        Ms.train(); C.train()
        total_ce = 0.0
        total_con = 0.0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            z = Ms(xb)
            logits = C(z)
            ce_loss = ce(logits, yb)
            con_loss = contrastive_loss(z, yb)
            loss = ce_loss + lambda_con * con_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_ce += ce_loss.item()
            total_con += con_loss.item()

        # -------- print source & target accuracy --------
        Ms.eval(); C.eval()
        with torch.no_grad():
            # source acc
            pred_s = C(Ms(X_s_tensor.to(device))).argmax(1).cpu()
            src_acc = (pred_s == y_s_tensor).float().mean().item()
            # target acc
            # pred_t = C(Ms(X_t_tensor.to(device))).argmax(1).cpu()
            # tgt_acc = (pred_t == y_t_tensor).float().mean().item()

        print(f"[Stage A][Epoch {ep}] CE={total_ce:.4f}  Con={total_con:.4f}  "
              f"SrcAcc={src_acc:.4f}")

    return Ms, C


# ========================= ADDA TRAIN =========================
def train_adda(Ms, C, src_loader, tgt_loader, epochs=20, lr_disc=1e-4, lr_mt=2e-4):

    Ms.eval()
    for p in Ms.parameters(): p.requires_grad_(False)
    C.eval()
    for p in C.parameters(): p.requires_grad_(False)

    Mt = Encoder(input_dim, hidden_dim=64, latent_dim=64).to(device)
    Mt.load_state_dict(Ms.state_dict())
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

            with torch.no_grad():
                f_s = Ms(x_s)

            f_t = Mt(x_t)

            # train D
            y_s = torch.ones(f_s.size(0), 1, device=device)
            y_t = torch.zeros(f_t.size(0), 1, device=device)
            log_s = D(f_s.detach())
            log_t = D(f_t.detach())
            loss_D = bce(log_s, y_s) + bce(log_t, y_t)
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # train Mt
            f_t = Mt(x_t)
            log_t2 = D(f_t)
            loss_Mt = bce(log_t2, torch.ones_like(y_t))
            opt_Mt.zero_grad()
            loss_Mt.backward()
            opt_Mt.step()

            total_d += loss_D.item()
            total_mt += loss_Mt.item()

        # -------- print epoch source & target accuracy --------
        Mt.eval(); C.eval()
        with torch.no_grad():
            # pred_s = C(Mt(X_s_tensor.to(device))).argmax(1).cpu()
            # src_acc = (pred_s == y_s_tensor).float().mean().item()

            pred_t = C(Mt(X_t_tensor.to(device))).argmax(1).cpu()
            tgt_acc = (pred_t == y_t_tensor).float().mean().item()

        print(f"[Stage B][Epoch {ep}] loss_D={total_d:.4f}  loss_Mt={total_mt:.4f}  "
              f"TgtAcc={tgt_acc:.4f}")

    return Mt


# ========================= MAIN =========================
if __name__ == "__main__":
    Ms = Encoder(input_dim, hidden_dim=64, latent_dim=64)
    C  = LabelClassifier(latent_dim=64, num_classes=num_classes)

    Ms, C = train_source(Ms, C, src_loader, epochs=20, lr=1e-3, lambda_con=0.2)
    Mt = train_adda(Ms, C, src_loader, tgt_loader, epochs=20)

    # ================= FINAL SAVE =================
    Mt.eval()
    with torch.no_grad():
        z_s = Mt(X_s_tensor.to(device)).cpu().numpy()
        z_t = Mt(X_t_tensor.to(device)).cpu().numpy()
        z_all = np.concatenate([z_s, z_t], axis=0)

        logits_s = C(Mt(X_s_tensor.to(device)))
        y_pred_s = logits_s.argmax(1).cpu().numpy()

        logits_t = C(Mt(X_t_tensor.to(device)))
        y_pred_t = logits_t.argmax(1).cpu().numpy()

        y_pred_all = np.concatenate([y_pred_s, y_pred_t], axis=0)

        np.save("../results/z_adv.npy", z_all)
        np.save("../results/y_pred_adv.npy", y_pred_all)

    # ================= FINAL ACC =================
    Mt.eval(); C.eval()
    with torch.no_grad():
        pred_s = C(Mt(X_s_tensor.to(device))).argmax(1).cpu()
        final_src_acc = (pred_s == y_s_tensor).float().mean().item()

        pred_t = C(Mt(X_t_tensor.to(device))).argmax(1).cpu()
        final_tgt_acc = (pred_t == y_t_tensor).float().mean().item()

    print("\n===== FINAL ACCURACY =====")
    print("Final Source Accuracy =", final_src_acc)
    print("Final Target Accuracy =", final_tgt_acc)
