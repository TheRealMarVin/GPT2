import argparse
import os

import torch
import torch.nn as nn
from datasets import load_dataset
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from torch_datasets.next_token_dataset import NextTokenDataset
from helpers.config_helpers import load_config
from models.gpt import GPT
from schedulers.warmup_cosine_scheduler import WarmupCosineScheduler
from train_eval.common_training_setup import TrainingConfig, run_specific_experiment
from utils.download_datasets import download_sherlock_dataset
from utils.next_token_training import next_token_train_epoch, next_token_evaluate


def main(training_config, model_config):
    print("Start Training")

    if "seed" in training_config:
        seed = training_config["seed"]
        torch.manual_seed(seed)

    model = GPT(model_config)

    if "clip_grad_norm" in training_config and training_config["clip_grad_norm"]:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    data_file = download_sherlock_dataset("data", "sherlock.txt")
    with open(data_file, "r", encoding="utf-8") as file:
        text_data = file.read()

    train_ratio = training_config["train_ratio"]
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    train_dataset = NextTokenDataset(txt=train_data, tokenizer=model.tokenizer, max_length=256, stride=128)

    test_data = text_data[split_idx:]
    test_dataset = NextTokenDataset(txt=test_data, tokenizer=model.tokenizer, max_length=256, stride=128)

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

    train_config = TrainingConfig(datasets=(train_dataset, test_dataset),
                                  batch_size=training_config["batch_size"],
                                  nb_epochs=nb_epochs,
                                  learning_rate=learning_rate,
                                  scheduler=scheduler,
                                  experiment_name=model_name,
                                  save_logic=save_logic,
                                  ignore_validation=False)

    train_args = {"optimizer": optimizer,
                  "criterion": nn.CrossEntropyLoss(),
                  "metrics_dict": {"loss": nn.CrossEntropyLoss()},
                  "model": model}
    eval_args = {"model": model, "metrics_dict": {"loss": nn.CrossEntropyLoss()}}

    run_specific_experiment(train_config, next_token_train_epoch, train_args,
                            eval_logic=next_token_evaluate, eval_args=eval_args,
                            summary=summary)

    start_context = "Sherlock entered the"
    print("Input text:", start_context)

    out = model.generate_text(contexts=start_context)
    print(out)

    print("Training Done!")


if __name__ == "__main__":
    default_model_config = "configs/gpt_124M_pre_trained.yaml"
    default_training_config = "configs/fine_tuning_training.yaml"

    parser = argparse.ArgumentParser(description="Configuration to launch training of the next token.")
    parser.add_argument("--model", type=str, default=default_model_config, help="Configuration to define the model to use during training.")
    parser.add_argument("--training", type=str, default=default_training_config, help="Configuration of the training routine.")
    args = parser.parse_args()

    model_config = load_config(args.model)
    training_config = load_config(args.training)
    main(training_config, model_config)