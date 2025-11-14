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

data = np.load("../data/frog_zeb_pca100.npz", allow_pickle=True)
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
    latent_dim = 64
    hidden_dim = 64
    lambda_d = 0.1
    lambda_contrastive = 0.2
    use_contrastive = True
    num_epochs = 20
    lr = 1e-3

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



        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Loss={avg_loss:.4f} | y={total_y_loss/len(train_loader):.4f} | "
              f"d={total_d_loss/len(train_loader):.4f} | con={total_con_loss/len(train_loader):.4f}")
        
    encoder.eval()
    label_clf.eval()
    with torch.no_grad():
        X_all_tensor = torch.tensor(X_all, dtype=torch.float32).to(device)
        z_all = encoder(X_all_tensor).cpu().numpy()
        logits_all = label_clf(encoder(X_all_tensor))
        y_pred_all = logits_all.argmax(1).cpu().numpy()
        np.save("../results/z_grl.npy", z_all)
        np.save("../results/y_pred_grl.npy", y_pred_all)
