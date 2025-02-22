import torch
import torch.nn as nn
from transformers import GPT2Tokenizer

from models.transformer_block import TransformerBlock


class GPTBackBone(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.model_name = config["model"]["name"]
        nb_layers = config["model"]["nb_layers"]
        self.tokenizer = GPT2Tokenizer.from_pretrained(config["model"]["tokenizer"])
        vocab_size = self.tokenizer.vocab_size
        embedding_dim = config["model"]["embedding_dim"]

        self.tokens = nn.Embedding(vocab_size, embedding_dim)
        self.transformers_blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(nb_layers)])


    def forward(self, x):
        attention_mask = x == self.pad_token

        x = self.tokens(x)

        for i, block in enumerate(self.transformers_blocks):
            x = block(x, attention_mask)

        return x
