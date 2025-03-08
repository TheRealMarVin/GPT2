import numpy as np
import torch
from transformers import GPT2Model


def load_weights(gpt, config):
    gpt_hf = GPT2Model.from_pretrained(config["hf_model_name"], cache_dir="pretrained")

    hf_state_dict = gpt_hf.state_dict()

    # Embedding layers
    _check_and_assign(gpt.positional_embeddings.weight, hf_state_dict["wpe.weight"])
    _check_and_assign(gpt.tokens.weight, hf_state_dict["wte.weight"])
    if "final_norm" in gpt.__dict__["_modules"]:
        _check_and_assign(gpt.final_norm.weight, hf_state_dict["ln_f.weight"])
        _check_and_assign(gpt.final_norm.bias, hf_state_dict["ln_f.bias"])
    if "out_layer" in gpt.__dict__["_modules"]:
        _check_and_assign(gpt.out_layer.weight, hf_state_dict["wte.weight"])

    # Transformer block layers
    for b in range(config["nb_layers"]):
        block_prefix = f"h.{b}."

        # Update transformer blocks
        _update_attention(hf_state_dict, block_prefix, gpt.transformers_blocks[b].attention_layer, config)
        _update_linear_layers(hf_state_dict, block_prefix, gpt.transformers_blocks[b].linear_block)
        _update_transformer_normalization_layers(hf_state_dict, block_prefix, gpt.transformers_blocks[b])


def _update_attention(hf_state_dict, block_prefix, attention_layer, config):
    q_w, k_w, v_w = np.split(hf_state_dict[f"{block_prefix}attn.c_attn.weight"], 3, axis=-1)
    q_b, k_b, v_b = np.split(hf_state_dict[f"{block_prefix}attn.c_attn.bias"], 3, axis=-1)
    _check_and_assign(attention_layer.query.weight, q_w.T)
    _check_and_assign(attention_layer.value.weight, v_w.T)
    _check_and_assign(attention_layer.key.weight, k_w.T)

    if config["attention"]["use_bias"]:
        _check_and_assign(attention_layer.query.bias, q_b)
        _check_and_assign(attention_layer.value.bias, v_b)
        _check_and_assign(attention_layer.key.bias, k_b)

    _check_and_assign(attention_layer.out_proj.weight, hf_state_dict[f"{block_prefix}attn.c_proj.weight"].T)
    _check_and_assign(attention_layer.out_proj.bias, hf_state_dict[f"{block_prefix}attn.c_proj.bias"])


def _update_linear_layers(hf_state_dict, block_prefix, linear_block):
    _check_and_assign(linear_block[1].weight, hf_state_dict[f"{block_prefix}mlp.c_fc.weight"].T)
    _check_and_assign(linear_block[1].bias, hf_state_dict[f"{block_prefix}mlp.c_fc.bias"])
    _check_and_assign(linear_block[3].weight, hf_state_dict[f"{block_prefix}mlp.c_proj.weight"].T)
    _check_and_assign(linear_block[3].bias, hf_state_dict[f"{block_prefix}mlp.c_proj.bias"])


def _update_transformer_normalization_layers(hf_state_dict, block_prefix, transformers_block):
    _check_and_assign(transformers_block.norm1.weight, hf_state_dict[f"{block_prefix}ln_1.weight"])
    _check_and_assign(transformers_block.norm1.bias, hf_state_dict[f"{block_prefix}ln_1.bias"])
    _check_and_assign(transformers_block.linear_block[0].weight, hf_state_dict[f"{block_prefix}ln_2.weight"])
    _check_and_assign(transformers_block.linear_block[0].bias, hf_state_dict[f"{block_prefix}ln_2.bias"])


def _check_and_assign(target, source):
    if target.shape != source.shape:
        raise ValueError(f"Shape mismatch. Target: {target.shape}, Source: {source.shape}")
    target.data.copy_(torch.nn.Parameter(source.clone().detach()))