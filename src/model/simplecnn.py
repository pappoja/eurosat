import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, num_non_image_features, num_countries, input_type='image', embedding_dim=16):
        super().__init__()
        self.input_type = input_type

        # 1–2 convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),  # [B, 32, 32, 32]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # [B, 64, 16, 16]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # [B, 64, 1, 1]
        )
        self.flatten = nn.Flatten()

        # Country embedding (optional)
        self.embedding_dim = embedding_dim
        if input_type in ['image_country', 'image_country_all']:
            self.country_embedding = nn.Embedding(num_embeddings=num_countries, embedding_dim=embedding_dim)

        # Non-image features (optional)
        if input_type == 'image_country_all':
            self.fc_non_image = nn.Linear(num_non_image_features, 32)

        # Add a linear layer after concatenation
        input_dim = 64
        if input_type in ['image_country', 'image_country_all']:
            input_dim += embedding_dim
        if input_type == 'image_country_all':
            input_dim += 32

        # Post-concatenation layer
        self.fc_post_concat = nn.Linear(input_dim, 128)

        # Final classifier
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x, features=None, country_idx=None):
        x = self.conv(x)
        x = self.flatten(x)

        if self.input_type in ['image_country', 'image_country_all'] and country_idx is not None:
            country_emb = self.country_embedding(country_idx)
        else:
            country_emb = torch.zeros((x.size(0), self.embedding_dim), device=x.device)

        if self.input_type == 'image_country_all' and features is not None:
            non_img_feat = torch.relu(self.fc_non_image(features))
        else:
            non_img_feat = torch.zeros((x.size(0), 32), device=x.device)

        if self.input_type == 'image':
            combined = x
        elif self.input_type == 'image_country':
            combined = torch.cat([x, country_emb], dim=1)
        else:  # 'image_country_all'
            combined = torch.cat([x, country_emb, non_img_feat], dim=1)

        # Pass through the new linear layer
        combined = torch.relu(self.fc_post_concat(combined))

        return self.fc(combined)