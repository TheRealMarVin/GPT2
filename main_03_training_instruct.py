import argparse
import json
import os
from functools import partial

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from helpers.config_helpers import load_config
from models.gpt import GPT
from schedulers.warmup_cosine_scheduler import WarmupCosineScheduler
from torch_datasets.alpaca_dataset import AlpacaDataset
from train_eval.common_training_setup import run_specific_experiment, TrainingConfig
from utils.download_instruct_dataset import load_or_download_instruct_dataset_file
from utils.instruction_helpers import format_input_for_alpaca, format_output_for_alpaca
from utils.lora_wrapper import adapt_model_for_lora
from utils.next_token_training import next_token_train_epoch, next_token_evaluate
from utils.training_utils import custom_collate_fn


def train_instruct(training_config, model_config, model=None, post_fix="", test_data=[]):
    print("Start Training")

    if "seed" in training_config:
        seed = training_config["seed"]
        torch.manual_seed(seed)

    if model is None:
        model = GPT(model_config)

    if "use_lora" in training_config and training_config["use_lora"]:
        adapt_model_for_lora(model, rank=16, alpha=16, ignore_list=["out_layer"])

    model.model_name = model.model_name + post_fix

    if "clip_grad_norm" in training_config and training_config["clip_grad_norm"]:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Before training")

    for instruction, _ in test_data:
        _display_sample(model, instruction, "")

    train_dataset, test_dataset = _create_alpaca_datasets(file_path="./data/instruction-data.json", model=model)

    learning_rate = training_config["learning_rate"]
    nb_epochs = training_config["nb_epochs"]
    model_name = model.model_name
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    def save_logic(save_path):
        if save_path is not None:
            save_file = os.path.join(save_path, model_name + ".pt")
            torch.save(model, save_file)

    summary = SummaryWriter()
    if "use_warmup" in training_config and training_config["use_warmup"]:
        scheduler = WarmupCosineScheduler(optimizer=optimizer, warmup_steps=10, total_steps=20,
                                          min_lr=0.0000001, summary=summary)
    else:
        scheduler = None

    customized_collate_fn = partial(custom_collate_fn, input_padding=model.tokenizer.eos_token_id,
                                    output_ignore_index=-100)
    train_config = TrainingConfig(datasets=(train_dataset, test_dataset),
                                  batch_size=training_config["batch_size"],
                                  nb_epochs=nb_epochs,
                                  learning_rate=learning_rate,
                                  scheduler=scheduler,
                                  experiment_name=model_name,
                                  save_logic=save_logic,
                                  ignore_validation=False,
                                  customized_collate_fn=customized_collate_fn)

    train_args = {"optimizer": optimizer,
                  "criterion": nn.CrossEntropyLoss(),
                  "metrics_dict": {"loss": nn.CrossEntropyLoss()},
                  "model": model}
    # eval_args = {"model": model, "metrics_dict": {"loss": nn.CrossEntropyLoss()}}

    run_specific_experiment(train_config, next_token_train_epoch, train_args,
                            eval_logic=next_token_evaluate,
                            summary=summary)

    for instruction, experiments in test_data:
        print("!!!!New Test Data!!!!")
        _display_sample(model, instruction, "", experiments=experiments)

    print("Training Done!")
    return model

def _create_alpaca_datasets(file_path, model, max_context_length=1024):
    tokenizer = model.tokenizer

    data = load_or_download_instruct_dataset_file(file_path)

    tokenized_data = []
    for entry in data:
        input = format_input_for_alpaca(entry)
        tokenized_input = tokenizer(input)["input_ids"]

        output = format_output_for_alpaca(entry)
        tokenized_output = tokenizer(output)["input_ids"]
        tokenized_output.append(tokenizer.eos_token_id)

        if len(tokenized_input) + len(tokenized_output) <= max_context_length:
            tokenized_data.append((tokenized_input, tokenized_output))

    train_data, test_data = train_test_split(tokenized_data, test_size=0.05, random_state=42)

    train_dataset = AlpacaDataset(train_data)
    test_dataset = AlpacaDataset(test_data)

    return train_dataset, test_dataset


def _display_sample(model, instruction, instruction_input, experiments=[]):
    print("********************************")
    entry = {"instruction": instruction, "input": instruction_input, "output": ""}
    start_context = format_input_for_alpaca(entry) + format_output_for_alpaca(entry)

    out = model.generate_text(contexts=start_context, eos_id=model.tokenizer.eos_token_id, remove_context=True)
    print("input: ", entry)
    print("output: ", out)

    for key, val, temperature in experiments:
        print("XXXX")
        print(f"Temperature: {temperature} - Sampling: {key}:{val} - Input text: {instruction} - ")
        if "top_k" in model.config["model"]["sampler"]:
            del model.config["model"]["sampler"]["top_k"]
        if "top_p" in model.config["model"]["sampler"]:
            del model.config["model"]["sampler"]["top_p"]

        model.config["model"]["sampler"][key] = val
        model.config["model"]["sampler"]["temperature"] = temperature

        out = model.generate_text(contexts=start_context, eos_id=model.tokenizer.eos_token_id, remove_context=True)
        print(out)


if __name__ == "__main__":
    default_model_config = "configs/gpt_355M_pre_trained.yaml"
    default_training_config = "configs/instruct_training.yaml"

    parser = argparse.ArgumentParser(description="Configuration to launch training instruction.")
    parser.add_argument("--model", type=str, default=default_model_config, help="Configuration to define the model to use during training.")
    parser.add_argument("--training", type=str, default=default_training_config, help="Configuration of the training routine.")
    args = parser.parse_args()

    model_config = load_config(args.model)
    training_config = load_config(args.training)

    experiments = [("top_k", 3, 1.0), ("top_k", 3, 0.25), ("top_k", 3, 2), ("top_k", 1, 1.0), ("top_k", 40, 1.0),
                   ("top_p", 0.1, 1.0), ("top_p", 0.1, 0.5), ("top_p", 0.1, 1.5), ("top_p", 0.05, 1.0),
                   ("top_p", 1.0, 1.0)]
    test_data = [("What is the color of the ocean?", []), ("Describe a sunny day.", []),
                 ("Where do Sherlock Holmes and Dr. Watson live?", experiments)]
    train_instruct(training_config, model_config, post_fix="_instruct", test_data=test_data)
