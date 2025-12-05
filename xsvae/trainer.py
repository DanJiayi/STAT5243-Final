import numpy as np
import torch
import torch.nn.functional as F

from config import (
    DEVICE,
    EPOCHS_ONEPHASE,
    EPOCHS_PHASE1,
    EPOCHS_PHASE2,
    RECON_W,
    CE_MAX_W,
    CTR_MAX_W,
    KL_MAX_W,
    MMD_MAX_W,
    SPECIES_MAX_W,
    PRINT_EVERY,
    LR
)

from losses import (
    kl_divergence,
    contrastive_loss,
    negative_binomial_loss,
    mse_reconstruction_loss,
    compute_mmd,
    reconstruction_loss
)

def train_one_phase(
    model, train_loader, epochs=EPOCHS_ONEPHASE,
    recon_w=RECON_W, ce_max=CE_MAX_W, ctr_max=CTR_MAX_W,
    kl_max=KL_MAX_W, mmd_max=MMD_MAX_W, species_max=SPECIES_MAX_W, lr=LR
):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):

        # ---- dynamic warmup ----
        kl_w  = min(1.0, epoch / 20.0) * kl_max
        ce_w  = min(1.0, epoch / 4.0) * ce_max
        ctr_w = min(1.0, epoch / 20.0) * ctr_max    # kept but will multiply 0 loss
        mmd_w = min(1.0, epoch / 15.0) * mmd_max
        species_w = min(1.0, epoch / 10.0) * species_max

        losses, recons, klds, ces, ctrs, mmds, specieses = [], [], [], [], [], [], []

        for batch in train_loader:
            x = batch["x"].to(DEVICE)
            s = batch["s"].to(DEVICE)
            y = batch["y"].to(DEVICE)

            optimizer.zero_grad()
            out = model(x, s)

            mean = out["mean"]
            mu = out["mu"]
            logvar = out["logvar"]
            z = out["z"]
            logits = out["logits"]

            # ---- loss 1: reconstruction ----
            recon = reconstruction_loss(x, mean, model)

            # ---- loss 2: KL ----
            kld = kl_divergence(mu, logvar)

            # ---- loss 3: CE (only source) ----
            src_mask = (s == 0)
            if src_mask.sum() > 0:
                ce = F.cross_entropy(logits[src_mask], y[src_mask])
            else:
                ce = torch.tensor(0.0, device=DEVICE)

            # ---- loss 4: contrastive (removed; keep 0 for logging) ----
            ctr = torch.tensor(0.0, device=DEVICE)

            # ---- loss 5: MMD alignment ----
            src_z = z[s == 0]
            tgt_z = z[s == 1]
            mmd = compute_mmd(src_z, tgt_z)

            # ---- loss 6: species prediction (optional) ----
            if model.use_species_pred and out["species_logits"] is not None:
                species = F.cross_entropy(out["species_logits"], s)
            else:
                species = torch.tensor(0.0, device=DEVICE)

            # total loss
            loss = (
                recon_w * recon +
                kl_w * kld +
                ce_w * ce +
                ctr_w * ctr +     # ctr=0
                mmd_w * mmd +
                species_w * species
            )

            if torch.isnan(loss):
                print("NaN in loss, stop training")
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            losses.append(loss.item())
            recons.append(recon.item())
            klds.append(kld.item())
            ces.append(ce.item())
            ctrs.append(ctr.item())
            mmds.append(mmd.item())
            specieses.append(species.item())

        if torch.isnan(loss):
            print("NaN in loss, stop training")
            break

        if epoch % PRINT_EVERY == 0:
            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"Loss={np.mean(losses):.4f} | "
                f"Recon={np.mean(recons):.4f} | "
                f"KLD={np.mean(klds):.4f} | "
                f"CE={np.mean(ces):.4f} | "
                f"CTR={np.mean(ctrs):.4f} | "
                f"MMD={np.mean(mmds):.4f} | "
                f"SPECIES={np.mean(specieses):.4f} | "
                f"[w: recon={recon_w}, KL={kl_w:.3f}, CE={ce_w:.3f}, "
                f"CTR={ctr_w:.3f}, MMD={mmd_w:.3f}, SPECIES={species_w:.3f}]"
            )

def train_two_phase(
    model,
    train_loader,
    epochs_phase1=EPOCHS_PHASE1,
    epochs_phase2=EPOCHS_PHASE2,
    recon_w=RECON_W,
    ce_max=CE_MAX_W,
    ctr_max=CTR_MAX_W,     # will remain but not used
    kl_max=KL_MAX_W,
    mmd_max=MMD_MAX_W,
    species_max=SPECIES_MAX_W,
    lr=LR
):
    print(f"\n===== Phase 1: Train pure VAE ({epochs_phase1} epochs) =====\n")

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs_phase1 + 1):

        kl_w = min(1.0, epoch / 10.0) * kl_max

        losses, recons, klds = [], [], []

        for batch in train_loader:
            x = batch["x"].to(DEVICE)
            s = batch["s"].to(DEVICE)     # no y for phase 1

            optimizer.zero_grad()
            out = model(x, s)

            mean = out["mean"]
            mu = out["mu"]
            logvar = out["logvar"]

            # ---- phase 1 loss (no CE, no MMD, no species) ----
            recon = reconstruction_loss(x, mean, model)
            kld = kl_divergence(mu, logvar)

            loss = recon_w * recon + kl_w * kld

            if torch.isnan(loss):
                print("NaN in Phase1, stop training")
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            losses.append(loss.item())
            recons.append(recon.item())
            klds.append(kld.item())

        if torch.isnan(loss):
            break

        print(
            f"Phase1 Epoch {epoch:03d}/{epochs_phase1} | "
            f"Loss={np.mean(losses):.4f} | Recon={np.mean(recons):.4f} | "
            f"KLD={np.mean(klds):.4f} | KL_w={kl_w:.3f}"
        )

    print(f"\n===== Phase 2: Train XSVAE ({epochs_phase2} epochs) =====\n")

    # freeze decoder
    for p in model.decoder.parameters():
        p.requires_grad = False

    # freeze NB dispersion
    if model.recon_mode == "nb":
        model.px_r.requires_grad = False

    # new optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    for epoch in range(1, epochs_phase2 + 1):

        kl_w  = min(1.0, epoch / 10.0) * kl_max
        ce_w  = min(1.0, epoch / 5.0)  * ce_max
        ctr_w = 0.0                     # contrastive disabled
        mmd_w = min(1.0, epoch / 12.0) * mmd_max
        species_w = min(1.0, epoch / 10.0) * species_max

        losses, recons, klds, ces, ctrs, mmds, specieses = (
            [], [], [], [], [], [], []
        )

        for batch in train_loader:
            x = batch["x"].to(DEVICE)
            s = batch["s"].to(DEVICE)
            y = batch["y"].to(DEVICE)

            optimizer.zero_grad()
            out = model(x, s)

            mean = out["mean"]
            mu = out["mu"]
            logvar = out["logvar"]
            z = out["z"]
            logits = out["logits"]

            # ---- VAE ----
            recon = reconstruction_loss(x, mean, model)
            kld = kl_divergence(mu, logvar)

            # ---- CE: only on source (no contrastive loss anymore) ----
            src_mask = (s == 0)
            if src_mask.sum() > 0:
                ce = F.cross_entropy(logits[src_mask], y[src_mask])
            else:
                ce = torch.tensor(0.0, device=DEVICE)

            ctr = torch.tensor(0.0, device=DEVICE)   # removed

            # ---- domain alignment (MMD) ----
            src_z = z[s == 0]
            tgt_z = z[s == 1]
            mmd = compute_mmd(src_z, tgt_z)

            # ---- species predictor (optional) ----
            if model.use_species_pred and out["species_logits"] is not None:
                species = F.cross_entropy(out["species_logits"], s)
            else:
                species = torch.tensor(0.0, device=DEVICE)

            # ---- final loss ----
            loss = (
                recon_w * recon +
                kl_w * kld +
                ce_w * ce +
                mmd_w * mmd +
                species_w * species
            )

            if torch.isnan(loss):
                print("NaN in Phase2, stop training")
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=5.0
            )
            optimizer.step()

            # logging
            losses.append(loss.item())
            recons.append(recon.item())
            klds.append(kld.item())
            ces.append(ce.item())
            ctrs.append(0.0)
            mmds.append(mmd.item())
            specieses.append(species.item())

        if torch.isnan(loss):
            break

        print(
            f"Epoch {epoch:03d}/{epochs_phase2} | "
            f"Loss={np.mean(losses):.4f} | Recon={np.mean(recons):.4f} | "
            f"KLD={np.mean(klds):.4f} | CE={np.mean(ces):.4f} | CTR=0.0000 | "
            f"MMD={np.mean(mmds):.4f} | SPECIES={np.mean(specieses):.4f} | "
            f"[w: recon={recon_w}, KL={kl_w:.3f}, CE={ce_w:.3f}, "
            f"MMD={mmd_w:.3f}, SPECIES={species_w:.3f}]"
        )