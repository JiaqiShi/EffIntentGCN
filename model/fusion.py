import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiFeatureFusion(nn.Module):
    def __init__(self, fusion_type, feature_dim, feature_num) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.fusion_type = fusion_type
        self.feature_num = feature_num

        if fusion_type == 'concat':
            self.fusion = DirectConcat(feature_dim, feature_num)
        elif fusion_type == 'weighted':
            self.fusion = WeightedFusion(feature_num)
        elif fusion_type == 'self_attention':
            self.fusion = SelfAttentionWithPosition(feature_dim, feature_num)
        elif fusion_type == 'self_attention_residual':
            self.fusion = SelfAttentionWithPositionResidual(feature_dim, feature_num)
        else:
            raise ValueError(f'[ERROR] Fusion type {fusion_type} does not exist.')
        
    def forward(self, *features):
        return self.fusion(*features)

class DirectConcat(nn.Module):
    def __init__(self, feature_dim, feature_num):
        super().__init__()
        self.fusion_layer = nn.Linear(feature_dim*feature_num, feature_dim)

    def forward(self, *features):
        return self.fusion_layer(torch.cat(features, dim=-1))

class WeightedFusion(nn.Module):
    def __init__(self, feature_num):
        super(WeightedFusion, self).__init__()
        self.weights = nn.Parameter(torch.ones(feature_num) / feature_num)

    def forward(self, *features):
        weighted_features = torch.stack(features, dim=-1)
        return torch.sum(self.weights[None, None, None, :] * weighted_features, dim=-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class SelfAttentionWithPosition(nn.Module):
    def __init__(self, feature_dim, feature_num):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(feature_dim, 8, batch_first=True)
        self.positional_encoder = PositionalEncoding(feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, *features):

        NT, V, C = features[0].shape
        features = [feature.view(NT*V, -1) for feature in features]

        x = torch.stack(features, dim=1)
        
        # Self attention and fusion
        x = self.positional_encoder(x) # NTV, num_modalities, C
        x, _ = self.attention(x, x, x)
        
        # Layer normalization
        x = self.norm(x)

        x = x[:, 0, :].view(NT, V, C)
        
        return x
    
class SelfAttentionWithPositionResidual(nn.Module):
    def __init__(self, feature_dim, feature_num):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(feature_dim, 8, batch_first=True)
        self.positional_encoder = PositionalEncoding(feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, *features):

        NT, V, C = features[0].shape
        features = [feature.view(NT*V, -1) for feature in features]

        features = torch.stack(features, dim=1)
        
        # Self attention and fusion
        x = self.positional_encoder(features) # NTV, num_modalities, C
        x, _ = self.attention(x, x, x)

        x_out = x + features
        
        # Layer normalization
        x = self.norm(x_out)

        x = x[:, 0, :].view(NT, V, C)
        
        return x