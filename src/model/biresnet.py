import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class BiResNet(nn.Module):
    def __init__(self, num_classes, num_non_image_features):
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
        self.num_resnet_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove the final layer
        
        # Additional layers for non-image data
        self.fc_non_image = nn.Linear(num_non_image_features, 128)
        self.fc_combined = nn.Linear(self.num_resnet_features + 128, num_classes)

    def forward(self, x, non_image_data):
        # Forward pass through ResNet
        x = self.resnet(x)
        
        # Forward pass through non-image data
        non_image_features = torch.relu(self.fc_non_image(non_image_data))
        
        # Concatenate image and non-image features
        combined_features = torch.cat((x, non_image_features), dim=1)
        
        # Final classification layer
        output = self.fc_combined(combined_features)
        return output
