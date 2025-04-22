import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, input_type='image', num_non_image_features=0, num_countries=0, embedding_dim=16):
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
            self.country_embedding = nn.Embedding(num_countries, embedding_dim)

        # Non-image features (optional)
        if input_type == 'image_country_all':
            self.fc_non_image = nn.Linear(num_non_image_features, 32)

        # Final classifier
        input_dim = 64
        if input_type in ['image_country', 'image_country_all']:
            input_dim += embedding_dim
        if input_type == 'image_country_all':
            input_dim += 32

        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x, country_idx=None, non_image_data=None):
        x = self.conv(x)
        x = self.flatten(x)

        if self.input_type in ['image_country', 'image_country_all'] and country_idx is not None:
            country_emb = self.country_embedding(country_idx)
        else:
            country_emb = torch.zeros((x.size(0), self.embedding_dim), device=x.device)

        if self.input_type == 'image_country_all' and non_image_data is not None:
            non_img_feat = torch.relu(self.fc_non_image(non_image_data))
        else:
            non_img_feat = torch.zeros((x.size(0), 32), device=x.device)

        if self.input_type == 'image':
            combined = x
        elif self.input_type == 'image_country':
            combined = torch.cat([x, country_emb], dim=1)
        else:  # 'image_country_all'
            combined = torch.cat([x, country_emb, non_img_feat], dim=1)

        return self.fc(combined)