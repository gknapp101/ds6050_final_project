import torch.nn as nn
from torchvision import models


class CNN_resnet_transfer(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

        for param in self.resnet.parameters():
            param.requires_grad = False

        for name, param in self.resnet.named_parameters():
            if "layer3" in name or "layer4" in name or "fc" in name:
                param.requires_grad = True

    def forward(self, X):
        return self.resnet(X)
