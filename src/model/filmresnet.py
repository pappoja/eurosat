import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights

resnet_variants = {
    'filmresnet18': (resnet18, ResNet18_Weights.DEFAULT),
    'filmresnet50': (resnet50, ResNet50_Weights.DEFAULT),
}

class FiLMResNet(nn.Module):
    def __init__(self, model_type, num_classes, num_non_image_features, num_countries, input_type='image_country_all', embedding_dim=16):
        super().__init__()
        assert model_type in resnet_variants, f"Unsupported model type: {model_type}"

        self.input_type = input_type
        self.embedding_dim = embedding_dim

        # Load base ResNet
        constructor, weights = resnet_variants[model_type]
        self.resnet = constructor(weights=weights)

        # Replace final classification layer with Identity
        self.num_resnet_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        # Country embedding
        self.country_embedding = nn.Embedding(num_countries, embedding_dim)

        # FiLM modulator (produces gamma and beta for FiLM layer)
        self.film_layer = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)  # 64 for gamma, 64 for beta
        )

        # Non-image data module
        self.fc_non_image = nn.Linear(num_non_image_features, 64)

        # Final classifier input size
        input_dim = self.num_resnet_features
        if input_type in ['image_country', 'image_country_all']:
            input_dim += embedding_dim
        if input_type == 'image_country_all':
            input_dim += 64

        self.fc_combined = nn.Linear(input_dim, num_classes)

    def forward(self, x, country_idx=None, non_image_data=None):
        # ----- Embed and modulate -----
        if self.input_type in ['image_country', 'image_country_all'] and country_idx is not None:
            country_emb = self.country_embedding(country_idx)
        else:
            country_emb = torch.zeros((x.size(0), self.embedding_dim), device=x.device)

        # Get FiLM parameters
        film_params = self.film_layer(country_emb)  # [B, 128]
        gamma, beta = film_params.chunk(2, dim=1)   # Each is [B, 64]

        # Apply early ResNet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        # Apply FiLM modulation after layer1
        x = self.resnet.layer1(x)

        # FiLM: scale & shift per channel
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, 64, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        x = gamma * x + beta

        # Continue with rest of ResNet
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)

        # Handle non-image data
        if self.input_type == 'image_country_all' and non_image_data is not None:
            non_img_feat = torch.relu(self.fc_non_image(non_image_data))
        else:
            non_img_feat = torch.zeros((x.size(0), 64), device=x.device)

        # Concatenate all modalities
        out = [x]
        if self.input_type in ['image_country', 'image_country_all']:
            out.append(country_emb)
        if self.input_type == 'image_country_all':
            out.append(non_img_feat)

        combined = torch.cat(out, dim=1)
        return self.fc_combined(combined)
