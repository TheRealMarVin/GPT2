def compute_trainable_params(model):
    total_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total_params


def freeze_weights(model, ignore_list=None):
    if ignore_list is None:
        ignore_list = []

    for name, param in model.named_parameters():
        if not any(name.startswith(layer + '.') for layer in ignore_list):
            param.requires_grad = False
