import argparse
import os
from functools import partial

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from torch_datasets.next_token_dataset import NextTokenDataset
from helpers.config_helpers import load_config
from models.gpt import GPT
from schedulers.warmup_cosine_scheduler import WarmupCosineScheduler
from train_eval.common_training_setup import TrainingConfig, run_specific_experiment
from utils.download_sherlock_datasets import download_sherlock_dataset
from utils.lora_wrapper import adapt_model_for_lora
from utils.next_token_training import next_token_train_epoch, next_token_evaluate
from utils.training_utils import custom_collate_fn


def train_next_token(training_config, model_config, post_fix="", experiments=[]):
    print("Start Training")

    if "seed" in training_config:
        seed = training_config["seed"]
        torch.manual_seed(seed)

    model = GPT(model_config)
    model.model_name = model.model_name + post_fix

    if "use_lora" in training_config and training_config["use_lora"]:
        adapt_model_for_lora(model, rank=16, alpha=16, ignore_list=["out_layer"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    start_context = "Where do Sherlock Holmes and Dr. Watson live?"

    out = model.generate_text(contexts=start_context)
    print("text before training:")
    print("Input text:", start_context)
    print(out)

    data_file = download_sherlock_dataset("data", "sherlock.txt")
    with open(data_file, "r", encoding="utf-8") as file:
        text_data = file.read()

    train_ratio = training_config["train_ratio"]
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    train_dataset = NextTokenDataset(txt=train_data, tokenizer=model.tokenizer, max_length=model_config["model"]["context_length"], stride=training_config["stride"])

    test_data = text_data[split_idx:]
    test_dataset = NextTokenDataset(txt=test_data, tokenizer=model.tokenizer, max_length=model_config["model"]["context_length"], stride=256)

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

    customized_collate_fn = partial(custom_collate_fn, input_padding=model.tokenizer.eos_token_id, output_ignore_index=-100)
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

    run_specific_experiment(train_config, next_token_train_epoch, train_args,
                            eval_logic=next_token_evaluate, summary=summary)

    out = model.generate_text(contexts=start_context)
    print("text after training:")
    print("Input text:", start_context)
    print(out)

    for key, val, temperature in experiments:
        print("*******************")
        print(f"Input text: {start_context} - Temperature: {temperature} - Sampling: {key}:{val}")
        if "top_k" in model.config["model"]["sampler"]:
            del model.config["model"]["sampler"]["top_k"]
        if "top_p" in model.config["model"]["sampler"]:
            del model.config["model"]["sampler"]["top_p"]

        model.config["model"]["sampler"][key] = val
        model.config["model"]["sampler"]["temperature"] = temperature

        out = model.generate_text(contexts=start_context)
        print(out)

    print("Training Done!")
    return model


if __name__ == "__main__":
    default_model_config = "configs/gpt_355M.yaml"
    default_training_config = "configs/pre_train_training.yaml"

    parser = argparse.ArgumentParser(description="Configuration to launch training of the next token.")
    parser.add_argument("--model", type=str, default=default_model_config, help="Configuration to define the model to use during training.")
    parser.add_argument("--training", type=str, default=default_training_config, help="Configuration of the training routine.")
    args = parser.parse_args()

    model_config = load_config(args.model)
    training_config = load_config(args.training)

    experiments = [("top_k", 3, 1.0), ("top_k", 3, 0.25), ("top_k", 3, 2), ("top_k", 1, 1.0), ("top_k", 40, 1.0),
                   ("top_p", 0.1, 1.0), ("top_p", 0.1, 0.5), ("top_p", 0.1, 1.5), ("top_p", 0.05, 1.0),
                   ("top_p", 1.0, 1.0)]

    train_next_token(training_config, model_config, experiments=experiments)