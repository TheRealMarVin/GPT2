import torch
import torch.nn as nn
from transformers import GPT2Tokenizer

from models.transformer_block import TransformerBlock
from utils.download_weights import load_weights


class GPTBackBone(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.pad_token = config["model"]["pad_token"]
        self.model_name = config["model"]["name"]
        nb_layers = config["model"]["nb_layers"]
        self.tokenizer = GPT2Tokenizer.from_pretrained(config["model"]["tokenizer"])
        vocab_size = self.tokenizer.vocab_size
        embedding_dim = config["model"]["embedding_dim"]

        self.tokens = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embeddings = nn.Embedding(config["model"]["context_length"], embedding_dim)
        self.embedding_dropout = nn.Dropout(config["model"]["token_dropout_rate"])

        self.transformers_blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(nb_layers)])

        self._init_extra_modules(config)

        if "hf_model_name" in config["model"]:
            load_weights(self, config["model"])

    def _init_extra_modules(self, config):
        pass

    def forward(self, x):
        batch_size, seq_len = x.shape
        attention_mask = x == self.pad_token

        x = self.tokens(x)
        position_embeddings = self.positional_embeddings(torch.arange(seq_len, device=x.device))
        x = x + position_embeddings
        x = self.embedding_dropout(x)


        for i, block in enumerate(self.transformers_blocks):
            x = block(x, attention_mask)

        return x
