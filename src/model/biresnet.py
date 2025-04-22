import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class BiResNet(nn.Module):
    def __init__(self, num_classes, num_non_image_features, num_countries, embedding_dim=16):
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
        self.resnet.fc = nn.Identity()
        
        # Embedding layer for country
        self.country_embedding = nn.Embedding(num_embeddings=num_countries, embedding_dim=embedding_dim)

        # Additional layers for non-image data
        self.fc_non_image = nn.Linear(num_non_image_features, 128)

        # Combine features
        self.fc_combined = nn.Linear(self.num_resnet_features + embedding_dim + 128, num_classes)

    def forward(self, x, country_idx=None, non_image_data=None):
        # Forward pass through ResNet
        resnet_features = self.resnet(x)
        
        # Forward pass through country embedding
        if country_idx is not None:
            embedded_country = self.country_embedding(country_idx)
        else:
            embedded_country = torch.zeros((x.size(0), self.country_embedding.embedding_dim), device=x.device)

        # Forward pass through non-image data
        if non_image_data is not None:
            non_image_features = torch.relu(self.fc_non_image(non_image_data))
        else:
            non_image_features = torch.zeros((x.size(0), 128), device=x.device)

        # Concatenate image, country, and non-image features
        combined_features = torch.cat((resnet_features, embedded_country, non_image_features), dim=1)

        # Final classification layer
        output = self.fc_combined(combined_features)
        return output
