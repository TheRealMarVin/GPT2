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
            mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
            self.register_buffer("mask", mask.bool())

    def forward(self, x, attention_mask=None):
        batch, context_length, embedding_dim = x.shape

        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        # This way of doing it in the book is quite cool
        keys = keys.view(batch, context_length, self.nb_heads, self.head_dim)
        values = values.view(batch, context_length, self.nb_heads, self.head_dim)
        queries = queries.view(batch, context_length, self.nb_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        if self.use_mask:
            mask_bool = self.mask[:context_length, :context_length]
            if attention_mask is not None:
                attention_mask = mask_bool | attention_mask.bool()
            else:
                attention_mask = mask_bool

        attn_scores.masked_fill_(attention_mask, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)

        context_vec = context_vec.contiguous().view(batch, context_length, self.embedding_dim)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec
