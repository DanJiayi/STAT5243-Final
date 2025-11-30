import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

# -----------------------------
# 1. load adata + Z embedding
# -----------------------------
key = "grl"
adata_filt = sc.read("../data/adata_filt.h5ad")

z = np.load(f"../results/z_{key}.npy")
y_pred = np.load(f"../results/y_pred_{key}.npy")

data = np.load("../data/frog_zeb_processed.npz", allow_pickle=True)
X_s = data["X_s"]; y_s = data["y_s"]
X_t = data["X_t"]; y_t = data["y_t"]
arr = data["label_map"]
label_map = {k: v for k, v in arr}
inv_label_map = {v: k for k, v in label_map.items()}

# -----------------------------
# 2. filter frog + zeb (保持原逻辑)
# -----------------------------
adata_s = adata_filt[adata_filt.obs["species"] == "frog"]
adata_t = adata_filt[adata_filt.obs["species"] == "zebrafish"]
adata_filt = adata_s.concatenate(adata_t, batch_key=None)

adata_filt.obsm["X_z"] = z

y_all = np.concatenate([y_s, y_t])
cell_types = np.array([inv_label_map[int(i)] for i in y_all])
adata_filt.obs["cell_type"] = cell_types

y_pred_names = np.array([inv_label_map[int(i)] for i in y_pred])
adata_filt.obs["y_pred"] = y_pred_names

# -----------------------------
# 3. neighbors + UMAP on z-space
# -----------------------------
sc.pp.neighbors(adata_filt, use_rep="X_z")
sc.tl.umap(adata_filt)

adata_t_z = adata_filt[adata_filt.obs["species"] == "zebrafish"].copy()

# -----------------------------
# 4. Original UMAP on X_t
# -----------------------------
adata_t_orig = sc.AnnData(X_t)
adata_t_orig.obs = adata_t_z.obs.copy()

sc.pp.neighbors(adata_t_orig)
sc.tl.umap(adata_t_orig)

# -----------------------------
# 5. 构建共享颜色（True label & Pred label 都必须一致）
# -----------------------------
all_cell_types = np.unique(cell_types)

from matplotlib import cm
cmap = cm.get_cmap("tab20", len(all_cell_types))
color_dict = {lab: cmap(i) for i, lab in enumerate(all_cell_types)}

# y_pred 的颜色必须与 y_true 对齐（按名字找颜色）
color_dict_pred = {
    lab: color_dict.get(lab, (0.5,0.5,0.5)) for lab in np.unique(y_pred_names)
}

# -----------------------------
# 6. 三张图并列 + 单一图例
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(22, 6))
# fig.subplots_adjust(top=0.88)

def plot_umap(ax, ad, key, title, palette):
    sc.pl.umap(
        ad,
        color=key,
        ax=ax,
        show=False,
        legend_loc=None,   # 不显示图例
        palette=palette,
        size=10,
        title=title
    )
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")

# 左：X_t (true labels)
plot_umap(
    axes[0],
    adata_t_orig,
    "cell_type",
    "PCA-space UMAP (ref labels)",
    color_dict
)

# 中：Z-space (true labels)
plot_umap(
    axes[1],
    adata_t_z,
    "cell_type",
    "Z-space UMAP (ref labels)",
    color_dict
)

# 右：Z-space (predicted labels)
plot_umap(
    axes[2],
    adata_t_z,
    "y_pred",
    "Z-space UMAP (pred labels)",
    color_dict_pred
)

# -----------------------------
# 7. 单一、共享图例
# -----------------------------
handles = [
    plt.Line2D([0], [0], marker='o', color=color_dict[lab],
               linestyle='', markersize=8)
    for lab in all_cell_types
]
labels = list(all_cell_types)

leg = fig.legend(
    handles, labels,
    title="Cell Types",
    loc="center right",
    fontsize=10,
    title_fontsize=11
)
leg.get_frame().set_linewidth(0)

plt.tight_layout(rect=[0, 0, 0.88, 1])
# fig.suptitle("UMAP on target species - annotated by true and predicted cell types", fontsize=18, y=1.02)
plt.savefig(f"../results/three_panel_umap_shared_legend_{key}.png", dpi=300)
plt.show()


# ==========================================================

import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

# =============================================================
# 1. Load all required data
# =============================================================
key = "grl"

# load filtered adata
adata_filt = sc.read("../data/adata_filt.h5ad")

# load learned embedding (full frog+zeb)
z = np.load(f"../results/z_{key}.npy")

# load predicted labels (not used here but loaded for completeness)
y_pred = np.load(f"../results/y_pred_{key}.npy")

# load processed frog + zeb original data
data = np.load("../data/frog_zeb_processed.npz", allow_pickle=True)
X_s = data["X_s"]               # frog original X
y_s = data["y_s"]
X_t = data["X_t"]               # zeb original X
y_t = data["y_t"]
arr = data["label_map"]
label_map     = {k: v for k, v in arr}
inv_label_map = {v: k for k, v in label_map.items()}

# =============================================================
# 2. Build full original X_all  (frog + zeb)
# =============================================================
X_all = np.concatenate([X_s, X_t], axis=0)

species_all = np.concatenate([
    np.array(["frog"] * len(X_s)),
    np.array(["zebrafish"] * len(X_t))
])

# Create AnnData for original feature space
adata_all_orig = sc.AnnData(X_all)
adata_all_orig.obs["species"] = species_all

# =============================================================
# 3. Build Z-space AnnData
# =============================================================
adata_all_z = adata_all_orig.copy()
adata_all_z.obsm["X_z"] = z

# =============================================================
# 4. Compute neighbors + UMAP
# =============================================================
# original X space
sc.pp.neighbors(adata_all_orig)
sc.tl.umap(adata_all_orig)

# z space
sc.pp.neighbors(adata_all_z, use_rep="X_z")
sc.tl.umap(adata_all_z)

# =============================================================
# 5. Shared colors for species
# =============================================================
species_list = ["frog", "zebrafish"]
from matplotlib import cm
cmap = cm.get_cmap("tab10", 2)
species_colors = {sp: cmap(i) for i, sp in enumerate(species_list)}

shuffle_idx = np.random.permutation(adata_all_orig.n_obs)
adata_all_orig = adata_all_orig[shuffle_idx].copy()
adata_all_z    = adata_all_z[shuffle_idx].copy()

# =============================================================
# 6. Plot two-panel UMAP with shared legend
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# fig.subplots_adjust(top=0.88)

def plot_umap_no_legend(ax, ad, title):
    sc.pl.umap(
        ad,
        color="species",
        palette=species_colors,
        ax=ax,
        show=False,
        legend_loc=None,
        size=10,
        title=title
    )
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")

# left: original X_all UMAP
plot_umap_no_legend(
    axes[0],
    adata_all_orig,
    "PCA-space UMAP (species)"
)

# right: z-space UMAP
plot_umap_no_legend(
    axes[1],
    adata_all_z,
    "Z-space UMAP (species)"
)

# shared legend on right
handles = [
    plt.Line2D([0], [0], marker='o', color=species_colors[sp],
               linestyle='', markersize=8)
    for sp in species_list
]
labels = species_list

leg = fig.legend(
    handles, labels,
    title="Species",
    loc="center right",
    fontsize=10,
    title_fontsize=11
)
leg.get_frame().set_linewidth(0)

plt.tight_layout(rect=[0, 0, 0.88, 1])
# fig.suptitle("UMAP on full data - annotated by species", fontsize=18, y=1.02)
plt.savefig(f"../results/two_panel_umap_species_{key}.png", dpi=300)
plt.show()

