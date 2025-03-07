import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_dim = config["model"]["embedding_dim"]
        self.nb_heads = config["model"]["attention"]["nb_heads"]
        self.head_dim = self.embedding_dim // self.nb_heads
        enable_bias = config["model"]["attention"]["use_bias"]
        self.use_mask = config["model"]["attention"]["use_mask"]
        dropout_rate = config["model"]["attention"]["dropout_rate"]

        self.query = nn.Linear(self.embedding_dim, self.embedding_dim, bias=enable_bias)
        self.key = nn.Linear(self.embedding_dim, self.embedding_dim, bias=enable_bias)
        self.value = nn.Linear(self.embedding_dim, self.embedding_dim, bias=enable_bias)

        self.out_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

        if self.use_mask:
            context_length = config["model"]["context_length"]
            causal_mask = torch.triu(torch.full((context_length, context_length), fill_value=-torch.inf), diagonal=1)
            self.register_buffer("mask", causal_mask)

    def forward(self, x, attention_mask=None):
        batch, context_length, embedding_dim = x.shape

        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        # Reshape for multi-head attention
        keys = keys.view(batch, context_length, self.nb_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(batch, context_length, self.nb_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch, context_length, self.nb_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(-2, -1)

        # Apply **causal mask** only during training
        if self.training and self.use_mask:
            causal_mask = self.mask[:context_length, :context_length]
            attn_scores = attn_scores + causal_mask.unsqueeze(0).unsqueeze(0)

        # Apply **padding mask** always
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask.unsqueeze(1).unsqueeze(2)

        attn_weights = torch.softmax(attn_scores / self.head_dim ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2).contiguous().view(batch, context_length, self.embedding_dim)
        context_vec = self.out_proj(context_vec)

        return context_vec
