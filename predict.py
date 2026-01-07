import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from dataset import HouseDataset
from models import FusionRegressor

TEST_PATH = Path("data/raw/test2.xlsx")
IMG_TEST_DIR = Path("data/images/test")

CKPT_PATH = Path("outputs/models/best_multimodal.pt")
OUT_PATH = Path("outputs/preds/submission.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_ckpt(path: Path):
    """
    Loads the checkpoint safely. If your torch version supports weights_only=True,
    we use it; otherwise we fall back to default.
    """
    try:
        # Newer torch versions support weights_only
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older torch versions: no weights_only argument
        return torch.load(path, map_location="cpu")
	


def main():
    # 1) Read test
    test_df = pd.read_excel(TEST_PATH)

    # 2) Load checkpoint
    ckpt = load_ckpt(CKPT_PATH)

    tab_cols = ckpt["tab_cols"]
    scaler_mean = ckpt["scaler_mean"]
    scaler_scale = ckpt["scaler_scale"]

    # 3) Ensure required columns exist in test
    missing = [c for c in ["id"] + tab_cols if c not in test_df.columns]
    if missing:
        raise ValueError(f"Missing columns in test: {missing}")

    # 4) Build a scaler that knows feature names (removes sklearn warning)
    scaler = StandardScaler()
    # Fit on a dummy DF with the same columns so scaler stores feature_names_in_
    dummy = pd.DataFrame(np.zeros((1, len(tab_cols))), columns=tab_cols)
    scaler.fit(dummy)

    # Overwrite learned params from training checkpoint
    scaler.mean_ = np.array(scaler_mean)
    scaler.scale_ = np.array(scaler_scale)
    scaler.n_features_in_ = len(tab_cols)
    scaler.feature_names_in_ = np.array(tab_cols)

    # 5) Transform test tabular cols
    test_df[tab_cols] = scaler.transform(test_df[tab_cols])

    # 6) Dataset + loader
    ds = HouseDataset(test_df, IMG_TEST_DIR, tab_cols, target_col=None)

    loader = DataLoader(
        ds,
        batch_size=32 if DEVICE == "cuda" else 16,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )

    # 7) Rebuild model and load weights
    model = FusionRegressor(tab_in=len(tab_cols)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 8) Predict
    ids, preds = [], []
    use_amp = (DEVICE == "cuda")

    with torch.no_grad():
        for img, tab, pid in loader:
            img = img.to(DEVICE, non_blocking=True)
            tab = tab.to(DEVICE, non_blocking=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    pred_log = model(img, tab).detach().cpu().numpy()
            else:
                pred_log = model(img, tab).detach().cpu().numpy()

            pred_price = np.expm1(pred_log)  # log->price
            ids.extend(pid)
            preds.extend(pred_price.reshape(-1))

    # 9) Save submission
    out = pd.DataFrame({"id": ids, "predicted_price": preds})
    out.to_csv(OUT_PATH, index=False)
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()
