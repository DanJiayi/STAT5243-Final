import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

# -----------------------------
# 1. load adata + z embedding
# -----------------------------
key = "grl"
adata_filt = sc.read("../data/adata_filt.h5ad")

z = np.load(f"../results/z_{key}.npy")
y_pred = np.load(f"../results/y_pred_{key}.npy")

data = np.load("../data/frog_zeb_pca100.npz", allow_pickle=True)
y_s = data["y_s"]
y_t = data["y_t"]
arr = data["label_map"]
label_map = {k: v for k, v in arr}
inv_label_map = {v: k for k, v in label_map.items()}

# -----------------------------
# 2. filter frog + zeb & reorder
# -----------------------------
adata_s = adata_filt[adata_filt.obs["species"] == "frog"]
adata_t = adata_filt[adata_filt.obs["species"] == "zebrafish"]
adata_filt = adata_s.concatenate(adata_t, batch_key=None)

adata_filt.obsm["X_z"] = z

# true label → name
y_all = np.concatenate([y_s, y_t])
cell_types = np.array([inv_label_map[int(i)] for i in y_all])
adata_filt.obs["cell_type"] = cell_types

# predicted label → name
y_pred_names = np.array([inv_label_map[int(i)] for i in y_pred])
adata_filt.obs["y_pred"] = y_pred_names

# -----------------------------
# 3. neighbors + UMAP
# -----------------------------
sc.pp.neighbors(adata_filt, use_rep="X_z")
sc.tl.umap(adata_filt)

adata_s = adata_filt[adata_filt.obs["species"] == "frog"]
adata_t = adata_filt[adata_filt.obs["species"] == "zebrafish"]

# -----------------------------
# 4. Make 6-panel UMAP figure (shared legend)
# -----------------------------
fig, axes = plt.subplots(3, 2, figsize=(14, 18))

# collect all cell types (ref + pred)
all_labels = np.unique(np.concatenate([
    adata_filt.obs["cell_type"].astype(str).values,
    adata_filt.obs["y_pred"].astype(str).values
]))

# assign colors
from matplotlib import cm
cmap = cm.get_cmap('tab20', len(all_labels))
color_dict = {lab: cmap(i) for i, lab in enumerate(all_labels)}

# helper to plot without legend
def plot_panel(ax, ad, color, title):
    sc.pl.umap(
        ad,
        color=color,
        ax=ax,
        show=False,
        title=title,
        size=20,
        legend_loc=None,   # IMPORTANT: disable per-panel legend
        palette=color_dict
    )

# Row 1: Source
plot_panel(axes[0,0], adata_s, "cell_type", "Source (ref_labels)")
plot_panel(axes[0,1], adata_s, "y_pred",    "Source (y_pred)")

# Row 2: Target
plot_panel(axes[1,0], adata_t, "cell_type", "Target (ref_labels)")
plot_panel(axes[1,1], adata_t, "y_pred",    "Target (y_pred)")

# Row 3: Combined
plot_panel(axes[2,0], adata_filt, "cell_type", "Combined (ref_labels)")
plot_panel(axes[2,1], adata_filt, "y_pred",    "Combined (y_pred)")

# -----------------------------
# 5. Add ONE unified legend on the right
# -----------------------------
handles = []
labels = []

for lab, col in color_dict.items():
    handles.append(plt.Line2D([0], [0], marker='o', color=col,
                              linestyle='', markersize=8))
    labels.append(lab)

fig.legend(
    handles,
    labels,
    title="Cell Types",
    loc="center right",
    bbox_to_anchor=(1.05, 0.5),
    fontsize=10,
    title_fontsize=11
)

plt.tight_layout(rect=[0, 0, 0.88, 1])
plt.savefig(f"../results/umap_compare_{key}.png", dpi=300)
plt.close()

print(f"Saved figure: ../results/umap_compare_{key}.png")
