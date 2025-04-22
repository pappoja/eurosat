import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights

resnet_variants = {
    'resnet18': (resnet18, ResNet18_Weights.DEFAULT),
    'resnet50': (resnet50, ResNet50_Weights.DEFAULT)
}

class ResNet(nn.Module):
    def __init__(self, model_type: str, num_classes: int):
        super().__init__()
        assert model_type in resnet_variants, f"Unsupported model type: {model_type}"
        
        constructor, weights = resnet_variants[model_type]
        self.resnet = constructor(weights=weights)
        
        # Replace first conv layer to accept 3 channels explicitly
        original_layer = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.resnet.conv1.weight.copy_(original_layer.weight)
        
        # Replace the final classification layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
