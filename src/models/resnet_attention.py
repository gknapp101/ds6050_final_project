import torch
import torch.nn as nn
from torchvision import models


class ChannelAttention(nn.Module):
    """Squeeze excitation style channel gate (shared MLP on avg + max pool)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al.)."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


class resnet_attention(nn.Module):
    """
    ResNet18 transfer learning with CBAM on the same stages you fine-tune (layer3, layer4).
    Early layers frozen; layer3, layer4, fc, and both CBAM modules are trainable.
    """

    def __init__(self, num_classes: int, reduction: int = 16):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.conv1   = backbone.conv1
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1  = backbone.layer1
        self.layer2  = backbone.layer2
        self.layer3  = backbone.layer3
        self.layer4  = backbone.layer4
        self.avgpool = backbone.avgpool

        # ResNet18: layer3 -> 256 ch, layer4 -> 512 ch
        self.cbam3 = CBAM(256, reduction=reduction)
        self.cbam4 = CBAM(512, reduction=reduction)

        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

        # Freeze all, then unfreeze layer3, layer4, cbam3, cbam4, fc
        for p in self.parameters():
            p.requires_grad = False
        for m in (self.layer3, self.layer4, self.fc, self.cbam3, self.cbam4):
            for p in m.parameters():
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.cbam3(x)
        x = self.layer4(x)
        x = self.cbam4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
