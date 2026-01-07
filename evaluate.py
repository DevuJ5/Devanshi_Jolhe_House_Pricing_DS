import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from dataset import HouseDataset
from models import FusionRegressor

TRAIN_PATH = Path("data/raw/train(1).xlsx")
IMG_TRAIN_DIR = Path("data/images/train")
CKPT_PATH = Path("outputs/models/best_multimodal.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    df = pd.read_excel(TRAIN_PATH)

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    tab_cols = ckpt["tab_cols"]

    # rebuild scaler correctly
    scaler = StandardScaler()
    dummy = pd.DataFrame(np.zeros((1, len(tab_cols))), columns=tab_cols)
    scaler.fit(dummy)
    scaler.mean_ = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]
    scaler.n_features_in_ = len(tab_cols)
    scaler.feature_names_in_ = np.array(tab_cols)

    df[tab_cols] = scaler.transform(df[tab_cols])

    ds = HouseDataset(df, IMG_TRAIN_DIR, tab_cols, target_col="price")
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    model = FusionRegressor(tab_in=len(tab_cols)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    y_true_log, y_pred_log = [], []

    with torch.no_grad():
        for img, tab, y in loader:
            img, tab = img.to(DEVICE), tab.to(DEVICE)
            pred = model(img, tab).cpu().numpy()
            y_true_log.append(y.numpy())
            y_pred_log.append(pred)

    y_true_log = np.concatenate(y_true_log)
    y_pred_log = np.concatenate(y_pred_log)

    # ---- metrics (log space) ----
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    mae_log = mean_absolute_error(y_true_log, y_pred_log)
    r2 = r2_score(y_true_log, y_pred_log)

    # ---- original price space ----
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    rmse_price = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_price = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # ---- threshold accuracy ----
    pct_err = np.abs((y_true - y_pred) / y_true)
    acc_10 = np.mean(pct_err <= 0.10) * 100
    acc_20 = np.mean(pct_err <= 0.20) * 100
    acc_30 = np.mean(pct_err <= 0.30) * 100

    print("\n===== MODEL PERFORMANCE =====")
    print(f"RMSE (log): {rmse_log:.4f}")
    print(f"MAE  (log): {mae_log:.4f}")
    print(f"R²   (log): {r2:.4f}")
    print(f"RMSE (price): {rmse_price:,.2f}")
    print(f"MAE  (price): {mae_price:,.2f}")
    print(f"MAPE (%): {mape:.2f}")
    print(f"Within ±10% accuracy: {acc_10:.2f}%")
    print(f"Within ±20% accuracy: {acc_20:.2f}%")
    print(f"Within ±30% accuracy: {acc_30:.2f}%")

if __name__ == "__main__":
    main()
