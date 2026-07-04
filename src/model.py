import torch
import torch.nn as nn
from torchvision import models

class CytoMorphClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(CytoMorphClassifier, self).__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT
        self.backbone = models.efficientnet_b0(weights=weights)
        
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
