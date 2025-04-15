import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Load ResNet-50 with latest weights
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Modify first conv layer to accept 13 channels instead of 3
        original_layer = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(13, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Initialize the new layer with the weights from the original layer
        with torch.no_grad():
            # Average the weights across the channel dimension and repeat 13 times
            new_weight = original_layer.weight.mean(dim=1, keepdim=True).repeat(1, 13, 1, 1)
            # Normalize the weights to maintain the magnitude
            new_weight = new_weight * (3/13)
            self.resnet.conv1.weight.copy_(new_weight)
        
        # Modify the final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)