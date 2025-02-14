import torch
import torch.nn as nn

from models.gpt_backbone import GPTBackBone


class GPT(GPTBackBone):
    def __init__(self, config):
        super().__init__(config)

        embedding_dim = config["model"]["embedding_dim"]
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.out_layer = nn.Linear(embedding_dim, self.tokenizer.vocab_size, bias=False)

    def forward(self, x):
        x = super().forward(x)

        x = self.final_norm(x)

        return self.out_layer(x)
