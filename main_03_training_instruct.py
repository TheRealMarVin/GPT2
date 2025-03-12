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
from utils.next_token_training import next_token_train_epoch, next_token_evaluate
from utils.training_utils import custom_collate_fn


def train_instruct(training_config, model_config, model=None):
    print("Start Training")

    if "seed" in training_config:
        seed = training_config["seed"]
        torch.manual_seed(seed)

    if model is None:
        model = GPT(model_config)

    model.model_name = model.model_name + "_instruct"

    if "clip_grad_norm" in training_config and training_config["clip_grad_norm"]:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Before training")
    _display_sample(model, "What is the color of the ocean?", "")
    _display_sample(model, "Describe a sunny day.", "")

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
    scheduler = WarmupCosineScheduler(optimizer=optimizer, warmup_steps=10, total_steps=20,
                                      min_lr=0.0000001, summary=summary)

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

    _display_sample(model, "What is the color of the ocean?", "")
    _display_sample(model, "Describe a sunny day.", "")

    print("Training Done!")
    return model

def _create_alpaca_datasets(file_path, model):
    tokenizer = model.tokenizer

    data = load_or_download_instruct_dataset_file(file_path)

    tokenized_data = []
    for entry in data:
        input = format_input_for_alpaca(entry)
        tokenized_input = tokenizer(input)["input_ids"]

        output = format_output_for_alpaca(entry)
        tokenized_output = tokenizer(output)["input_ids"]
        tokenized_output.append(tokenizer.eos_token_id)

        if len(tokenized_input) + len(tokenized_output) <= 1024:
            tokenized_data.append((tokenized_input, tokenized_output))

    train_data, test_data = train_test_split(tokenized_data, test_size=0.05, random_state=42)

    train_dataset = AlpacaDataset(train_data)
    test_dataset = AlpacaDataset(test_data)

    return train_dataset, test_dataset


def _display_sample(model, instruction, instruction_input):
    entry = {"instruction": instruction, "input": instruction_input, "output": ""}
    start_context = format_input_for_alpaca(entry) + format_output_for_alpaca(entry)

    out = model.generate_text(contexts=start_context, eos_id=model.tokenizer.eos_token_id, remove_context=True)

    print("********************************")
    print("input: ", entry)
    print("output: ", out)
    out = None


if __name__ == "__main__":
    default_model_config = "configs/gpt_124M_pre_trained.yaml"
    default_training_config = "configs/instruct_training.yaml"

    parser = argparse.ArgumentParser(description="Configuration to launch training of the next token.")
    parser.add_argument("--model", type=str, default=default_model_config, help="Configuration to define the model to use during training.")
    parser.add_argument("--training", type=str, default=default_training_config, help="Configuration of the training routine.")
    args = parser.parse_args()

    model_config = load_config(args.model)
    training_config = load_config(args.training)
    train_instruct(training_config, model_config)