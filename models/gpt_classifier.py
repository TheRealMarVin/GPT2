import torch
import torch.nn as nn

from models.gpt_backbone import GPTBackBone


class GPTClassifier(GPTBackBone):
    def __init__(self):
        super.__init__()

    def forward(self, x):
        pass
