import torch.nn as nn

from models.lora import LinearWithLoRA
from utils.models_helpers import freeze_weights, compute_trainable_params


def adapt_model_for_lora(model, rank, alpha, ignore_list=[], verbose=True):
    if verbose:
        print("Trainable parameters before LoRA {}".format(compute_trainable_params(model)))

    freeze_weights(model, ignore_list=ignore_list)
    if verbose:
        print("Trainable parameters after freeze LoRA {}".format(compute_trainable_params(model)))

    _replace_linear_with_lora(model, rank, alpha, ignore_list=ignore_list)
    if verbose:
        print("Trainable parameters after LoRA {}".format(compute_trainable_params(model)))


def _replace_linear_with_lora(model, rank, alpha, ignore_list=[]):
    for name, module in model.named_children():
        if name in ignore_list:
            continue

        if isinstance(module, nn.Linear):
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            _replace_linear_with_lora(module, rank, alpha, ignore_list)
