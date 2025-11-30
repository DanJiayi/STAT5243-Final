import os
import re
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models import *
from torch.autograd import Function

log_file = open("../logs/log_grl_ablation.txt", "w")
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

# ============================
# 2. 读数据（和你原来一模一样）
# ============================
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
y_t_dummy = -np.ones_like(y_t, dtype=np.int64)  # prevent data leakage
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X_s.shape[1]
num_classes = len(np.unique(y_s))

# ============================
# 3. 用“你原来的训练代码”封装成函数
# ============================
def run_training_once(
    latent_dim,
    hidden_dim,
    lambda_d,
    lambda_contrastive,
    use_contrastive,
    num_epochs,
    lr
):
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

            # 1) Encoder
            z = encoder(xb)

            # 2) classification loss (source only)
            logits_y = label_clf(z)
            src_mask = (yb != -1)
            if src_mask.any():
                y_loss = criterion_y(logits_y[src_mask], yb[src_mask])
            else:
                y_loss = torch.tensor(0.0, device=device)

            # 3) domain loss (GRL)
            z_rev = grad_reverse(z, lambd=1.0)
            logits_d = domain_clf(z_rev).squeeze(1)
            d_loss = criterion_d(logits_d, db)

            # 4) supervised contrastive loss (source only)
            if use_contrastive and src_mask.sum() > 1:
                z_src = z[src_mask]
                y_src = yb[src_mask]
                con_loss = contrastive_loss(z_src, y_src)
            else:
                con_loss = torch.tensor(0.0, device=device)

            # 5) total loss
            loss = y_loss + lambda_d * d_loss + lambda_contrastive * con_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_y_loss += y_loss.item()
            total_d_loss += d_loss.item()
            total_con_loss += con_loss.item()

        avg_loss = total_loss / len(train_loader)

        # epoch acc（和你原来一样）
        encoder.eval()
        label_clf.eval()
        with torch.no_grad():
            z_s = encoder(X_s_tensor.to(device))
            logits_s = label_clf(z_s)
            y_pred_s = logits_s.argmax(1).cpu()
            src_acc = (y_pred_s == y_s_tensor).float().mean().item()

            z_t = encoder(X_t_tensor.to(device))
            logits_t = label_clf(z_t)
            y_pred_t = logits_t.argmax(1).cpu()
            tgt_acc = (y_pred_t == y_t_tensor).float().mean().item()

        print(
            f"[Train lambda_d={lambda_d:.3f}, use_con={use_contrastive}] "
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Loss={avg_loss:.4f} | y={total_y_loss/len(train_loader):.4f} | "
            f"d={total_d_loss/len(train_loader):.4f} | con={total_con_loss/len(train_loader):.4f} | "
            f"SrcAcc={src_acc:.4f} | TgtAcc={tgt_acc:.4f}"
        )

    # ---- Final evaluation（和你原来最终计算一样）----
    encoder.eval()
    label_clf.eval()
    with torch.no_grad():
        z_s = encoder(X_s_tensor.to(device))
        logits_s = label_clf(z_s)
        src_acc_final = (logits_s.argmax(1).cpu() == y_s_tensor).float().mean().item()

        z_t = encoder(X_t_tensor.to(device))
        logits_t = label_clf(z_t)
        tgt_acc_final = (logits_t.argmax(1).cpu() == y_t_tensor).float().mean().item()

    print("\n>>> FINAL ACC (this run) <<<")
    print(f"Source Acc = {src_acc_final:.4f}")
    print(f"Target Acc = {tgt_acc_final:.4f}")

    return (
        src_acc_final,
        tgt_acc_final,
        encoder.state_dict(),
        label_clf.state_dict(),
        domain_clf.state_dict()
    )

# ============================
# 4. 扫描 ../models 下的 checkpoint
# ============================
import os
import re

# model_dir = "../models"
# pattern = re.compile(
#     r"([0-9.]+)_([0-9.]+)_([0-9]+)_([0-9.]+)_([0-9]+)_([0-9]+)_([0-9.]+)_([0-9.]+)\.pth"
# )

# selected = []
# seen_configs = set()  # 用于去重 (latent, hidden, lambda_d, lambda_con)

# for fname in os.listdir(model_dir):
#     m = pattern.match(fname)
#     if not m:
#         continue

#     tgt_acc = float(m.group(1))
#     src_acc = float(m.group(2))
#     epoch = int(m.group(3))
#     lr = float(m.group(4))
#     latent_dim = int(m.group(5))
#     hidden_dim = int(m.group(6))
#     lambda_d = float(m.group(7))
#     lambda_con = float(m.group(8))

#     # ------------ 去重 based on back parameters ------------
#     config_key = (epoch, lr,latent_dim, hidden_dim, lambda_d, lambda_con)
#     if config_key in seen_configs:
#         print(f"[Skip duplicate] {fname}")
#         continue
#     seen_configs.add(config_key)
#     # ----------------------------------------------------

    # if tgt_acc > 0.81:
    #     selected.append((fname, tgt_acc, src_acc, epoch, lr,
    #                      latent_dim, hidden_dim, lambda_d, lambda_con))
    #     print(f"[Select] {fname}")

# print(f"\nFound {len(selected)} models with tgt_acc > 0.81")

import os
import re

model_dir = "../models"

# 文件名格式:
# tgt_acc_src_acc_epoch_lr_latent_hidden_lambda_d_lambda_con.pth
pattern = re.compile(
    r"([0-9.]+)_([0-9.]+)_([0-9]+)_([0-9A-Za-z.+-]+)_([0-9]+)_([0-9]+)_([0-9.]+)_([0-9.]+)\.pth"
)

selected = []
seen_configs = set()

for fname in os.listdir(model_dir):

    m = pattern.match(fname)
    if not m:
        continue

    tgt_acc = float(m.group(1))
    src_acc = float(m.group(2))
    epoch = int(m.group(3))
    lr_str = m.group(4)            # ----★ 不转 float，保留原字符串 ----
    latent_dim = int(m.group(5))
    hidden_dim = int(m.group(6))
    lambda_d = float(m.group(7))
    lambda_con = float(m.group(8))

    # ----------- ★ 关键修改：要求 lr 必须是 "5e-05" ----------
    if lr_str != "5e-05":
        print(f"[Skip - lr != 5e-05] {fname}")
        continue
    # -------------------------------------------------------

    # ----------- 去重：后面相同参数组合略过 ----------
    config_key = (epoch, lr_str, latent_dim, hidden_dim, lambda_d, lambda_con)
    if config_key in seen_configs:
        print(f"[Skip duplicate] {fname}")
        continue
    seen_configs.add(config_key)
    # -------------------------------------------------------

    if tgt_acc > 0.81:
        selected.append(
            (fname, tgt_acc, src_acc, epoch, lr_str,
             latent_dim, hidden_dim, lambda_d, lambda_con)
        )
        print(f"[Select] {fname}")

print(f"\nFound {len(selected)} models with tgt_acc > 0.81 and lr=='5e-05'")



# ============================
# 5. 对每个模型做两种 ablation 重训
# ============================
for fname, tgt_acc_orig, src_acc_orig, epoch, lr, latent_dim, hidden_dim, lambda_d_orig, lambda_con_orig in selected:
    lr = 5e-5
    print("\n========================================")
    print(f"Processing model file: {fname}")
    print(f"orig tgt_acc={tgt_acc_orig:.4f}, src_acc={src_acc_orig:.4f}, "
          f"epoch={epoch}, lr={lr}, latent={latent_dim}, hidden={hidden_dim}, "
          f"lambda_d={lambda_d_orig}, lambda_con={lambda_con_orig}")
    print("========================================\n")

    flag = 1

    # ------------------- Ablation 1: lambda_d = 0, 仍然使用对比损失 -------------------
    print(">>> Ablation 1: lambda_d = 0, use_contrastive = True")
    src_acc_1, tgt_acc_1, enc_1, clf_1, dom_1 = run_training_once(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        lambda_d=0.0,
        lambda_contrastive=lambda_con_orig,
        use_contrastive=True,
        num_epochs=epoch,       # 路径里的 epoch -> num_epochs
        lr=lr
    )
    print('ACC:', src_acc_1, ',', tgt_acc_1)
    if not (round(tgt_acc_1, 4) <= round(tgt_acc_orig, 4)-0.002):
        print(">>> Ablation 1 tgt_acc NOT significantly lower than original, stopping further ablations for this model.")
        flag = 0

    # ------------------- Ablation 2: 去掉对比学习, lambda_d 用原来的 -------------------
    if flag==1:
        print("\n>>> Ablation 2: lambda_d = original, use_contrastive = False")
        src_acc_2, tgt_acc_2, enc_2, clf_2, dom_2 = run_training_once(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            lambda_d=lambda_d_orig,
            lambda_contrastive=lambda_con_orig,
            use_contrastive=False,
            num_epochs=epoch,
            lr=lr
        )
        print('ACC:', src_acc_2, ',', tgt_acc_2)
        if not (round(tgt_acc_2, 4) <= round(tgt_acc_orig, 4)-0.002):
            print(">>> Ablation 2 tgt_acc NOT significantly lower than original, skipping Ablation 3.")
            flag = 0

    if flag==1:
        # ------------------- Ablation 3------------------
        print("\n>>> Ablation 3: lambda_d = 0, use_contrastive = False")
        src_acc_3, tgt_acc_3, enc_3, clf_3, dom_3 = run_training_once(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            lambda_d=0.0,
            lambda_contrastive=lambda_con_orig,
            use_contrastive=False,
            num_epochs=epoch,
            lr=lr
        )
        print('ACC:', src_acc_3, ',', tgt_acc_3)

        # ------------------- 比较 + 保存 checkpoint -------------------
        print("\n>>> Compare with original")
        print(f"Original tgt_acc = {tgt_acc_orig:.4f}")
        print(f"Ablation1 tgt_acc = {tgt_acc_1:.4f}")
        print(f"Ablation2 tgt_acc = {tgt_acc_2:.4f}")
        print(f"Ablation3 tgt_acc = {tgt_acc_3:.4f}")

        if round(tgt_acc_1, 4) <= round(tgt_acc_orig, 4)-0.002 and round(tgt_acc_2, 4) <= round(tgt_acc_orig, 4)-0.002:
            new_name = (
                f"abl_{tgt_acc_orig:.4f}_{src_acc_orig:.4f}_"
                f"{tgt_acc_1:.4f}_{src_acc_1:.4f}_"
                f"{tgt_acc_2:.4f}_{src_acc_2:.4f}_"
                f"{tgt_acc_3:.4f}_{src_acc_3:.4f}_"
                f"{epoch}_{lr}_{latent_dim}_{hidden_dim}_{lambda_d_orig}_{lambda_con_orig}.pth"
            )
            save_path = os.path.join(model_dir, new_name)
            torch.save(
                {
                    "enc_case1": enc_1,
                    "clf_case1": clf_1,
                    "dom_case1": dom_1,
                    "enc_case2": enc_2,
                    "clf_case2": clf_2,
                    "dom_case2": dom_2,
                    "enc_case3": enc_3,
                    "clf_case3": clf_3,
                    "dom_case3": dom_3,
                    "meta": {
                        "orig_tgt": tgt_acc_orig,
                        "orig_src": src_acc_orig,
                        "case1": {"tgt": tgt_acc_1, "src": src_acc_1},
                        "case2": {"tgt": tgt_acc_2, "src": src_acc_2},
                        "case3": {"tgt": tgt_acc_3, "src": src_acc_3},
                        "epoch": epoch,
                        "lr": lr,
                        "latent_dim": latent_dim,
                        "hidden_dim": hidden_dim,
                        "lambda_d_orig": lambda_d_orig,
                        "lambda_con_orig": lambda_con_orig,
                    },
                },
                save_path,
            )
            print(f"\n*** Saved ablation checkpoint to: {save_path} ***")
        else:
            print("\nNo save: at least one ablation tgt_acc >= original tgt_acc.")

print("\n===== All done =====")
