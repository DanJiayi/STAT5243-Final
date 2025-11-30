import os
import re
import pandas as pd

model_dir = "/Users/jiayidan/Desktop/projects/bio/final project/STAT5243-Final-Project/models"

# 正则匹配 abl_ 开头的完整格式
pattern = re.compile(
    r"abl_"
    r"([0-9.]+)_"   # tgt_orig
    r"([0-9.]+)_"   # src_orig
    r"([0-9.]+)_"   # tgt1
    r"([0-9.]+)_"   # src1
    r"([0-9.]+)_"   # tgt2
    r"([0-9.]+)_"   # src2
    r"([0-9.]+)_"   # tgt3
    r"([0-9.]+)_"   # src3
    r"([0-9]+)_"    # epoch
    r"([0-9.+-eE]+)_"   # lr
    r"([0-9]+)_"    # latent_dim
    r"([0-9]+)_"    # hidden_dim
    r"([0-9.]+)_"   # lambda_d
    r"([0-9.]+)\.pth"  # lambda_con
)

rows = []

for fname in os.listdir(model_dir):
    if not fname.startswith("abl_"):
        continue

    m = pattern.match(fname)
    if not m:
        print(f"[Skip] Not matching pattern: {fname}")
        continue

    try:
        (
            tgt_orig, src_orig,
            tgt1, src1,
            tgt2, src2,
            tgt3, src3,
            epoch, lr,
            latent_dim, hidden_dim,
            lambda_d, lambda_con
        ) = m.groups()

        # 转成 float/int
        tgt_orig = float(tgt_orig)
        src_orig = float(src_orig)
        tgt1 = float(tgt1)
        src1 = float(src1)
        tgt2 = float(tgt2)
        src2 = float(src2)
        tgt3 = float(tgt3)
        src3 = float(src3)

        epoch = int(epoch)
        lr = float(lr)
        latent_dim = int(latent_dim)
        hidden_dim = int(hidden_dim)
        lambda_d = float(lambda_d)
        lambda_con = float(lambda_con)

    except Exception as e:
        print(f"[Skip] Cannot parse numbers in: {fname}")
        continue

    rows.append({
        "file": fname,
        "tgt_acc_orig": tgt_orig,
        "tgt_acc_1": tgt1,
        "tgt_acc_2": tgt2,
        "tgt_acc_3": tgt3,
        "orig - 1": tgt_orig - tgt1,
        "orig - 2": tgt_orig - tgt2,
        "orig - 3": tgt_orig - tgt3,
        "src_acc_orig": src_orig,
        "src_acc_1": src1,
        "src_acc_2": src2,
        "src_acc_3": src3,
        "epoch": epoch,
        "lr": lr,
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "lambda_d": lambda_d,
        "lambda_con": lambda_con,
    })

df = pd.DataFrame(rows)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

print("\n==================== ABLATION TABLE ====================\n")
print(df)
print("\n========================================================\n")
df.to_csv("../results/grl_ablation_results.csv", index=False)
