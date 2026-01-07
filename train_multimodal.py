# src/train_multimodal.py

import os
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dataset import HouseDataset
from models import FusionRegressor

TRAIN_PATH = Path("data/raw/train(1).xlsx")
IMG_TRAIN_DIR = Path("data/images/train")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_cnn(model):
    # Works for your FusionRegressor where CNN is model.cnn
    for p in model.cnn.parameters():
        p.requires_grad = False


def unfreeze_last_cnn_params(model, n_params=0):
    """
    If n_params==0: unfreeze all CNN params (slowest, best fine-tune).
    If n_params>0: unfreeze only last n_params tensors (fast + accurate compromise).
    """
    params = list(model.cnn.parameters())
    for p in params:
        p.requires_grad = False
    if n_params == 0:
        for p in params:
            p.requires_grad = True
    else:
        for p in params[-n_params:]:
            p.requires_grad = True


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    set_seed(42)

    # GPU speed knobs
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True

    df = pd.read_excel(TRAIN_PATH)

    drop_cols = ["price", "id"]
    tab_cols = [c for c in df.columns if c not in drop_cols]
    tab_cols = [c for c in tab_cols if np.issubdtype(df[c].dtype, np.number)]

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    train_df[tab_cols] = scaler.fit_transform(train_df[tab_cols])
    val_df[tab_cols] = scaler.transform(val_df[tab_cols])

    train_ds = HouseDataset(train_df, IMG_TRAIN_DIR, tab_cols, target_col="price")
    val_ds = HouseDataset(val_df, IMG_TRAIN_DIR, tab_cols, target_col="price")

    # ---- DataLoader (fast on GPU) ----
    # Start safe defaults for Windows; increase workers if stable
    # RTX 3050 4GB: batch_size=16/24 usually safe. We'll pick 16 by default.
    BATCH_SIZE = 16 if DEVICE == "cuda" else 32
    NUM_WORKERS = 2 if DEVICE == "cuda" else 0  # on GPU, 2 helps feeding the GPU

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    print("Train batches:", len(train_loader), flush=True)
    print("Val batches:", len(val_loader), flush=True)
    print("Device:", DEVICE, flush=True)
    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    model = FusionRegressor(tab_in=len(tab_cols)).to(DEVICE)

    use_amp = (DEVICE == "cuda")
    scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)
    loss_fn = torch.nn.MSELoss()

    best_val = 1e18
    best_path = Path("outputs/models/best_multimodal.pt")
    best_path.parent.mkdir(parents=True, exist_ok=True)

    PRINT_EVERY = 50 if DEVICE == "cuda" else 20  # less spam on GPU

    # =========================
    # Stage 1: Train head only (fast warmup)
    # =========================
    freeze_cnn(model)
    print("\nStage 1: frozen CNN")
    print("Trainable params:", count_trainable(model), flush=True)

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-3,  # higher LR for head
        weight_decay=1e-4
    )

    stage1_epochs = 2

    for epoch in range(1, stage1_epochs + 1):
        print(f"\nStarting epoch {epoch}/{stage1_epochs} (Stage 1)", flush=True)
        model.train()
        train_losses = []

        for batch_idx, (img, tab, y) in enumerate(train_loader, start=1):
            img = img.to(DEVICE, non_blocking=True)
            tab = tab.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    pred = model(img, tab)
                    loss = loss_fn(pred, y)
                scaler_amp.scale(loss).backward()
                scaler_amp.step(opt)
                scaler_amp.update()
            else:
                pred = model(img, tab)
                loss = loss_fn(pred, y)
                loss.backward()
                opt.step()

            train_losses.append(float(loss.item()))

            if (batch_idx % PRINT_EVERY == 0) or (batch_idx == len(train_loader)):
                print(
                    f"  Stage1 Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss {loss.item():.4f}",
                    flush=True
                )

        # validate each epoch
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for img, tab, y in val_loader:
                img = img.to(DEVICE, non_blocking=True)
                tab = tab.to(DEVICE, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(img, tab).detach().cpu().numpy()
                ys.append(y.numpy())
                ps.append(pred)

        val_rmse = rmse(np.concatenate(ys), np.concatenate(ps))
        print(f"Stage1 Epoch {epoch} | TrainLoss {np.mean(train_losses):.4f} | ValRMSE(log) {val_rmse:.4f}", flush=True)

        if val_rmse < best_val:
            best_val = val_rmse
            torch.save(
                {
                    "model": model.state_dict(),
                    "scaler_mean": scaler.mean_,
                    "scaler_scale": scaler.scale_,
                    "tab_cols": tab_cols,
                },
                best_path,
            )
            print("  Saved best:", best_path, flush=True)

    # =========================
    # Stage 2: Fine-tune last CNN layers (accuracy)
    # =========================
    print("\nStage 2: fine-tune last CNN layers (fast + accurate)")
    # Unfreeze only last few parameter tensors for speed.
    # If you want maximum accuracy and time is OK, set n_params=0 (unfreeze all CNN).
    unfreeze_last_cnn_params(model, n_params=30 if DEVICE == "cuda" else 0)
    print("Trainable params:", count_trainable(model), flush=True)

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-4,  # lower LR for finetuning
        weight_decay=1e-4
    )

    stage2_epochs = 6  # total 8 epochs. Increase to 8 if you have time.

    for e in range(1, stage2_epochs + 1):
        epoch = stage1_epochs + e
        print(f"\nStarting epoch {epoch}/{stage1_epochs + stage2_epochs} (Stage 2)", flush=True)

        model.train()
        train_losses = []

        for batch_idx, (img, tab, y) in enumerate(train_loader, start=1):
            img = img.to(DEVICE, non_blocking=True)
            tab = tab.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    pred = model(img, tab)
                    loss = loss_fn(pred, y)
                scaler_amp.scale(loss).backward()
                scaler_amp.step(opt)
                scaler_amp.update()
            else:
                pred = model(img, tab)
                loss = loss_fn(pred, y)
                loss.backward()
                opt.step()

            train_losses.append(float(loss.item()))

            if (batch_idx % PRINT_EVERY == 0) or (batch_idx == len(train_loader)):
                print(
                    f"  Stage2 Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss {loss.item():.4f}",
                    flush=True
                )

        # validate
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for img, tab, y in val_loader:
                img = img.to(DEVICE, non_blocking=True)
                tab = tab.to(DEVICE, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(img, tab).detach().cpu().numpy()
                ys.append(y.numpy())
                ps.append(pred)

        val_rmse = rmse(np.concatenate(ys), np.concatenate(ps))
        print(f"Epoch {epoch:02d} | TrainLoss {np.mean(train_losses):.4f} | ValRMSE(log) {val_rmse:.4f}", flush=True)

        if val_rmse < best_val:
            best_val = val_rmse
            torch.save(
                {
                    "model": model.state_dict(),
                    "scaler_mean": scaler.mean_,
                    "scaler_scale": scaler.scale_,
                    "tab_cols": tab_cols,
                },
                best_path,
            )
            print("  Saved best:", best_path, flush=True)

    print("\nBest Val RMSE(log):", best_val, flush=True)
    print("Best checkpoint:", best_path, flush=True)


if __name__ == "__main__":
    main()
