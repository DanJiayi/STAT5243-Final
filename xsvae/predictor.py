import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report

from config import DEVICE

def predict_on_loader(model, loader, id2label=None):
    """
    Return:
        acc      : accuracy
        y_true   : true labels
        y_pred   : predicted labels
    """
    model.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(DEVICE)
            s = batch["s"].to(DEVICE)
            y = batch["y"].cpu().numpy()

            out = model(x, s)
            logits = out["logits"].detach().cpu().numpy()
            pred = np.argmax(logits, axis=1)

            all_true.append(y)
            all_pred.append(pred)

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    acc = accuracy_score(y_true, y_pred)

    if id2label is not None:
        print(classification_report(
            [id2label[int(i)] for i in y_true],
            [id2label[int(i)] for i in y_pred],
            zero_division=0
        ))

    return acc, y_true, y_pred


def evaluate_model(model, source_loader, target_loader, id2label=None):
    acc_s, _, _ = predict_on_loader(model, source_loader, id2label)
    print("Source Acc =", acc_s)

    acc_t, _, _ = predict_on_loader(model, target_loader, id2label)
    print("Target Acc =", acc_t)

    return acc_s, acc_t

def compute_CI(acc_list):
    acc_arr = np.array(acc_list)
    mean = acc_arr.mean()
    std = acc_arr.std(ddof=1)
    ci95 = 1.96 * std / np.sqrt(len(acc_arr))
    return mean, ci95