import numpy as np
import torch
from torch.utils.data import DataLoader

from config import BATCH_SIZE, NPZ_PATH

def to_dense(x):
    return np.asarray(x.todense()) if hasattr(x, "todense") else np.asarray(x)


class FrogZebDataset(torch.utils.data.Dataset):
    def __init__(self, X, species, labels):
        self.X = X.astype(np.float32)
        self.species = species.astype(np.int64)
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {
            "x": self.X[idx],
            "s": self.species[idx],
            "y": self.labels[idx]
        }


def load_frog_zeb_data(
    npz_path: str = NPZ_PATH,
    batch_size: int = BATCH_SIZE,
    use_target_in_train: bool = True,
):
    data = np.load(npz_path, allow_pickle=True)

    X_s = data["X_s"]
    y_s = data["y_s"]
    X_t = data["X_t"]
    y_t = data["y_t"]

    print("Loaded:")
    print("  Source X_s:", X_s.shape, " labels:", y_s.shape)
    print("  Target X_t:", X_t.shape, " labels:", y_t.shape)

    X_s = to_dense(X_s)
    X_t = to_dense(X_t)
    X = np.concatenate([X_s, X_t], axis=0).astype(np.float32)
    species = np.concatenate([
        np.zeros(len(X_s), dtype=int),
        np.ones(len(X_t), dtype=int)
    ])
    labels = np.concatenate([y_s, y_t]).astype(int)

    n_cells, n_features = X.shape
    n_labels = len(np.unique(labels))

    print(f"\nTotal cells: {n_cells}, features: {n_features}")
    print(f"Source cells: {len(X_s)}, Target cells: {len(X_t)}")
    print(f"Num classes: {n_labels}")

    # label map
    if "label_map" in data.files:
        raw_lm = data["label_map"]
        id2label = {int(row[1]): str(row[0]) for row in raw_lm}
    else:
        id2label = {i: f"class_{i}" for i in np.unique(labels)}

    print("\nLabel map:")
    for k in list(id2label.keys())[:8]:
        print(f"  {k}: {id2label[k]}")

    full_dataset = FrogZebDataset(X, species, labels)

    # Source / Target
    src_idx = np.where(species == 0)[0]
    tgt_idx = np.where(species == 1)[0]

    src_dataset = torch.utils.data.Subset(full_dataset, src_idx)
    tgt_dataset = torch.utils.data.Subset(full_dataset, tgt_idx)

    if use_target_in_train:
        print("Training mode: source + target (XSVAE full)")
        train_dataset = full_dataset
    else:
        print("Training mode: source ONLY (no target data used)")
        train_dataset = src_dataset

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    src_loader_eval = DataLoader(src_dataset, batch_size=batch_size, shuffle=False)
    tgt_loader_eval = DataLoader(tgt_dataset, batch_size=batch_size, shuffle=False)

    print("\nDataLoader ready:")
    print(f"  Train loader: {len(train_loader)} batches")
    print(f"  Source eval loader: {len(src_loader_eval)} batches")
    print(f"  Target eval loader: {len(tgt_loader_eval)} batches")

    return train_loader, src_loader_eval, tgt_loader_eval, n_features, n_labels, id2label
