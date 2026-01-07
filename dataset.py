import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class HouseDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: Path, tabular_cols, target_col=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.tabular_cols = tabular_cols
        self.target_col = target_col

        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # [0,1]
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row["id"]

        img_path = self.image_dir / f"{pid}.png"
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
        else:
            img = Image.new("RGB", (224, 224), color=(0, 0, 0))

        img = self.tf(img)

        tab = row[self.tabular_cols].values.astype(np.float32)
        tab = torch.from_numpy(tab)

        if self.target_col is not None:
            y = np.log1p(float(row[self.target_col]))
            y = torch.tensor(y, dtype=torch.float32)
            return img, tab, y

        return img, tab, pid
