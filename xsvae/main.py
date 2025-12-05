from typing import Callable, List

from config import (
    DEVICE,
    RECON_MODE,
    HIDDEN_DIMS,
    LATENT_DIM,
    EPOCHS_ONEPHASE,
    EPOCHS_PHASE1,
    EPOCHS_PHASE2,
    RECON_W,
    CE_MAX_W,
    CTR_MAX_W,
    KL_MAX_W,
    MMD_MAX_W,
    SPECIES_MAX_W,
    LR,
)
from model import XSVAE
from dataloader import load_frog_zeb_data
from trainer import train_one_phase, train_two_phase
from predictor import evaluate_model, compute_CI
from utils import set_seed


def single_run_example(
    use_two_phase: bool = True,
    use_species_gate: bool = True,
    use_species_pred: bool = False,
):
    train_loader, src_loader_eval, tgt_loader_eval, n_features, n_labels, id2label = \
        load_frog_zeb_data()

    print("\n========== Initialize Model ==========\n")
    model = XSVAE(
        input_dim=n_features,
        hidden_dims=HIDDEN_DIMS,
        latent_dim=LATENT_DIM,
        n_species=2,
        n_labels=n_labels,
        recon_mode=RECON_MODE,
        use_species_gate=use_species_gate,
        use_species_pred=use_species_pred,
    ).to(DEVICE)

    print(model)
    print("\nModel loaded on", DEVICE)

    if use_two_phase:
        train_two_phase(
            model,
            train_loader,
            epochs_phase1=EPOCHS_PHASE1,
            epochs_phase2=EPOCHS_PHASE2,
            recon_w=RECON_W,
            ce_max=CE_MAX_W,
            ctr_max=CTR_MAX_W,
            kl_max=KL_MAX_W,
            mmd_max=MMD_MAX_W,
            species_max=SPECIES_MAX_W,
            lr=LR,
        )
    else:
        train_one_phase(
            model,
            train_loader,
            epochs=EPOCHS_ONEPHASE,
            recon_w=RECON_W,
            ce_max=CE_MAX_W,
            ctr_max=CTR_MAX_W,
            kl_max=KL_MAX_W,
            mmd_max=MMD_MAX_W,
            species_max=SPECIES_MAX_W,
            lr=LR,
        )

    print("\n========== Final Evaluation ==========\n")
    acc_s, acc_t = evaluate_model(model, src_loader_eval, tgt_loader_eval, id2label)

    print("\n========== Final Accuracies ==========")
    print(f"Source Acc = {acc_s:.4f}")
    print(f"Target Acc = {acc_t:.4f}")

    return acc_s, acc_t


def run_multi_seed_experiments(
    num_runs: int = 20,
    train_fn: Callable = train_one_phase,     # train_two_phase
    seeds: List[int] = None,
    use_two_phase: bool = True,
    use_species_gate: bool = True,
    use_species_pred: bool = False,
):
    if seeds is None:
        seeds = list(range(1, num_runs + 1))

    source_acc_list = []
    target_acc_list = []

    train_loader, src_loader_eval, tgt_loader_eval, n_features, n_labels, id2label = \
        load_frog_zeb_data()

    for i, seed in enumerate(seeds, start=1):
        print("\n\n==============================")
        print(f" Run {i}/{len(seeds)}   (seed={seed})")
        print("==============================")

        set_seed(seed)

        # --- reinitialize model ---
        model = XSVAE(
            input_dim=n_features,
            hidden_dims=HIDDEN_DIMS,
            latent_dim=LATENT_DIM,
            n_species=2,
            n_labels=n_labels,
            recon_mode=RECON_MODE,
            use_species_gate=use_species_gate,
            use_species_pred=use_species_pred,
        ).to(DEVICE)

        print(model)
        print("\nModel loaded on", DEVICE)

        # --- train ---
        if train_fn is train_two_phase or use_two_phase:
            train_two_phase(
                model,
                train_loader,
                epochs_phase1=EPOCHS_PHASE1,
                epochs_phase2=EPOCHS_PHASE2,
                recon_w=RECON_W,
                ce_max=CE_MAX_W,
                ctr_max=CTR_MAX_W,
                kl_max=KL_MAX_W,
                mmd_max=MMD_MAX_W,
                species_max=SPECIES_MAX_W,
                lr=LR,
            )
        else:
            train_one_phase(
                model,
                train_loader,
                epochs=EPOCHS_ONEPHASE,
                recon_w=RECON_W,
                ce_max=CE_MAX_W,
                ctr_max=CTR_MAX_W,
                kl_max=KL_MAX_W,
                mmd_max=MMD_MAX_W,
                species_max=SPECIES_MAX_W,
                lr=LR,
            )

        # --- evaluate ---
        acc_s, acc_t = evaluate_model(model, src_loader_eval, tgt_loader_eval, id2label)

        source_acc_list.append(acc_s)
        target_acc_list.append(acc_t)

    print("\n========================================")
    print("        Multi-run Accuracy Stats        ")
    print("========================================")

    src_mean, src_ci = compute_CI(source_acc_list)
    tgt_mean, tgt_ci = compute_CI(target_acc_list)

    print(f"Source Acc: mean={src_mean:.4f}, 95% CI=±{src_ci:.4f}")
    print(f"Target Acc: mean={tgt_mean:.4f}, 95% CI=±{tgt_ci:.4f}")

    return {
        "source_acc": source_acc_list,
        "target_acc": target_acc_list,
        "source_mean": src_mean,
        "source_std": float((sum((x - src_mean) ** 2 for x in source_acc_list) / (len(source_acc_list)-1)) ** 0.5),
        "target_mean": tgt_mean,
        "target_std": float((sum((x - tgt_mean) ** 2 for x in target_acc_list) / (len(target_acc_list)-1)) ** 0.5),
        "source_ci": src_ci,
        "target_ci": tgt_ci,
    }


if __name__ == "__main__":
    # Example
    set_seed(0)
    single_run_example(
        use_two_phase=False,
        use_species_gate=True,
        use_species_pred=False,
    )

    # result = run_multi_seed_experiments(
    #     num_runs=20,
    #     train_fn=train_one_phase,
    #     use_two_phase=True,
    #     use_species_gate=True,
    #     use_species_pred=False,
    # )
