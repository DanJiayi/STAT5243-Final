import torch
import torch.nn.functional as F

# Loss functions（NB / MSE / CE / CTR / MMD / KL）

# 1. Negative Binomial Loss
def negative_binomial_loss(x, mu, r, eps=1e-8):
    """
    x, mu, r: (B, D)
    NB log-likelihood used in scVI:
      NB(x | mu, r)

    return: mean loss over batch
    """
    mu = torch.clamp(mu, min=1e-4)
    r = torch.clamp(r, min=1e-4)

    if r.dim() == 1:
        r = r.view(1, -1).expand_as(x)

    t1 = torch.lgamma(x + r) - torch.lgamma(r) - torch.lgamma(x + 1.0)
    t2 = r * (torch.log(r + eps) - torch.log(r + mu + eps))
    t3 = x * (torch.log(mu + eps) - torch.log(r + mu + eps))

    log_prob = t1 + t2 + t3  # (B, D)
    return -log_prob.sum(dim=1).mean()  # scalar


# 2. MSE Reconstruction Loss
def mse_reconstruction_loss(x, recon):
    """
    x, recon: (B, D)
    """
    return F.mse_loss(recon, x)


# 3. KL Divergence
def kl_divergence(mu, logvar):
    """
    KL(N(mu, sigma) || N(0,1))
    mu, logvar: (B, latent_dim)
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()


# 4. Supervised Contrastive Loss（InfoNCE）
def contrastive_loss(z, labels, temperature=0.1):
    """
    z: (B, d) embedding
    labels: (B,)
    Only source domain uses this. 
    """
    if z.size(0) < 2:
        return torch.tensor(0.0, device=z.device)

    z = F.normalize(z, dim=1)
    sim = z @ z.T / temperature    # (B, B)

    labels = labels.view(-1, 1)
    mask = (labels == labels.T).float().to(z.device)
    mask.fill_diagonal_(0)

    # denominator
    logsumexp = torch.logsumexp(sim, dim=1)

    # positive similarity sum
    pos_sim = (sim * mask).sum(dim=1)
    pos_count = mask.sum(dim=1)

    valid = pos_count > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=z.device)

    pos_mean = torch.zeros_like(pos_sim)
    pos_mean[valid] = pos_sim[valid] / pos_count[valid]

    loss = -(pos_mean[valid] - logsumexp[valid])
    return loss.mean()


# 5. Maximum Mean Discrepancy (RBF kernel)
def compute_mmd(x, y, sigma=1.0):
    """
    x, y: (Ns, D), (Nt, D)
    RBF kernel MMD for domain alignment.
    """
    if x.size(0) == 0 or y.size(0) == 0:
        return torch.tensor(0.0, device=x.device)

    xx = x @ x.T
    yy = y @ y.T
    xy = x @ y.T

    rx = xx.diag().unsqueeze(0)
    ry = yy.diag().unsqueeze(0)

    Kxx = torch.exp(-(rx.T + rx - 2 * xx) / (2 * sigma**2))
    Kyy = torch.exp(-(ry.T + ry - 2 * yy) / (2 * sigma**2))
    Kxy = torch.exp(-(rx.T + ry - 2 * xy) / (2 * sigma**2))

    mmd = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
    return mmd


# 6. Reconstruction Loss
def reconstruction_loss(x, mean, model):
    """
    model.recon_mode：
      - negative binomial loss
      - MSE loss
    """
    if model.recon_mode == "nb":
        r = F.softplus(model.px_r)   # (D,)
        return negative_binomial_loss(x, mean, r)
    else:  # "mse"
        return mse_reconstruction_loss(x, mean)
