import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights

resnet_variants = {
    'biresnet18': (resnet18, ResNet18_Weights.DEFAULT),
    'biresnet50': (resnet50, ResNet50_Weights.DEFAULT),
}

class BiResNet(nn.Module):
    def __init__(self, model_type, num_classes, num_non_image_features, num_countries, input_type='image_country_all', embedding_dim=16):
        super().__init__()
        assert model_type in resnet_variants, f"Unsupported model type: {model_type}"

        # Load base ResNet
        constructor, weights = resnet_variants[model_type]
        self.resnet = constructor(weights=weights)

        # Replace initial conv layer (for compatibility)
        original_layer = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.resnet.conv1.weight.copy_(original_layer.weight)

        # Replace final classification layer with Identity to get features
        self.num_resnet_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        # Embedding for country
        self.country_embedding = nn.Embedding(num_embeddings=num_countries, embedding_dim=embedding_dim)

        # Feed-forward for non-image data
        self.fc_non_image = nn.Linear(num_non_image_features, 64)

        # Configuration for what inputs to use
        self.input_type = input_type

        # Final fully connected based on input type
        input_dim = self.num_resnet_features
        if input_type in ['image_country', 'image_country_all']:
            input_dim += embedding_dim
        if input_type == 'image_country_all':
            input_dim += 64

        # Add a linear layer after concatenation
        # self.fc_post_concat = nn.Linear(input_dim, 128)  

        # Final fully connected layer
        self.fc_combined = nn.Linear(input_dim, num_classes) # change input_dim to 128 for post-concat

    def forward(self, x, country_idx=None, non_image_data=None):
        resnet_features = self.resnet(x)

        if self.input_type in ['image_country', 'image_country_all'] and country_idx is not None:
            embedded_country = self.country_embedding(country_idx)
        else:
            embedded_country = torch.zeros((x.size(0), self.country_embedding.embedding_dim), device=x.device)

        if self.input_type == 'image_country_all' and non_image_data is not None:
            non_image_features = torch.relu(self.fc_non_image(non_image_data))
        else:
            non_image_features = torch.zeros((x.size(0), 64), device=x.device)

        if self.input_type == 'image':
            combined = resnet_features
        elif self.input_type == 'image_country':
            combined = torch.cat([resnet_features, embedded_country], dim=1)
        else:
            combined = torch.cat([resnet_features, embedded_country, non_image_features], dim=1)

        # Pass through the new linear layer
        #combined = torch.relu(self.fc_post_concat(combined))

        return self.fc_combined(combined)