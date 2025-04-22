import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet18, ResNet18_Weights

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Load ResNet-50 with latest weights
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Modify first conv layer to accept 3 channels (RGB)
        original_layer = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Initialize the new layer with the weights from the original layer
        with torch.no_grad():
            # Use the original weights directly for 3 channels
            self.resnet.conv1.weight.copy_(original_layer.weight)
        
        # Modify the final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Load ResNet-18 with latest weights
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Modify first conv layer to accept 3 channels (RGB)
        original_layer = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Initialize the new layer with the weights from the original layer
        with torch.no_grad():
            # Use the original weights directly for 3 channels
            self.resnet.conv1.weight.copy_(original_layer.weight)
        
        # Modify the final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)