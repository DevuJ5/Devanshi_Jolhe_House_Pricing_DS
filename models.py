import torch
import torch.nn as nn
import torchvision.models as models
import timm



class TabularMLP(nn.Module):
    def __init__(self, in_features: int, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class FusionRegressor(nn.Module):
    def __init__(self, tab_in: int, tab_emb_dim=128):
        super().__init__()

        # EfficientNet-B0 backbone (good accuracy/compute)
        self.cnn = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
        img_emb_dim = self.cnn.num_features  # embedding size

        self.tab_mlp = TabularMLP(tab_in, hidden=tab_emb_dim)

        self.head = nn.Sequential(
            nn.Linear(img_emb_dim + tab_emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 1)
        )

    def forward(self, img, tab):
        img_feat = self.cnn(img)
        tab_feat = self.tab_mlp(tab)
        fused = torch.cat([img_feat, tab_feat], dim=1)
        out = self.head(fused).squeeze(1)
        return out
