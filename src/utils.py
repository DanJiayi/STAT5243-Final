import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models import *
from torch.autograd import Function

def evaluate(encoder, classifier, loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.eval(); classifier.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            z = encoder(xb)
            logits = classifier(z)
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return correct / total