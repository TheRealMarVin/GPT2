import torch
import torch.nn as nn

from models.transformer_block import TransformerBlock


class GPTBackBone(nn.Module):
    def __init__(self, config):
        self.name = config["model"]["name"]
        nb_layers = config["model"]["nb_layers"]
        vocab_size = config["model"]["vocab_size"] # TODO... I guess it is not the right way
        embedding_dim = config["model"]["embedding_dim"]

        self.tokens = nn.Embedding(vocab_size, embedding_dim)
        self.transformers_blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(nb_layers)])


    def forward(self, x):
        x = self.tokens(x)

        for i, block in enumerate(self.transformers_blocks):
            x = block(x)

        return x
