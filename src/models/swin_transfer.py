import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights


class swin_transfer(nn.Module):
    """
    Swin-T transfer learning for SAR ship classification.
    Early stages frozen; stage 3, stage 4, norm, and head are trainable.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        backbone = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)

        self.features = backbone.features
        self.norm     = backbone.norm

        # Swin-T output dim is 768
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, num_classes)
        )

        # Freeze all first
        for p in self.parameters():
            p.requires_grad = False

        # Unfreeze last two stages, norm, and head
        for m in (self.features[6], self.features[7], self.norm, self.head):
            for p in m.parameters():
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)     # (B, H, W, C) — Swin is channels last
        x = self.norm(x)         # (B, H, W, C)
        x = x.mean(dim=[1, 2])  # (B, C=768)   — global avg pool over H, W
        return self.head(x)      # (B, num_classes)
