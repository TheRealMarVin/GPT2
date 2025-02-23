import torch
import torch.nn as nn

from models.multi_head_attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        embedding_dim = config["model"]["embedding_dim"]
        dropout_rate = config["model"]["transformer_block"]["dropout_rate"]

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.attention_layer = MultiHeadAttention(config)

        expand_factor = config["model"]["transformer_block"]["expand_factor"]
        self.linear_block = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, expand_factor * embedding_dim),
            nn.GELU(),
            nn.Linear(expand_factor * embedding_dim, embedding_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x, attention_mask):
        shortcut = x
        x = self.norm1(x)
        x = self.attention_layer(x, attention_mask)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.linear_block(x)
        x = x + shortcut

        return x
