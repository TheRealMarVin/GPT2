import os

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from datasets.next_token_dataset import NextTokenDataset
from helpers.config_helpers import load_config
from models.gpt import GPT
from schedulers.warmup_cosine_scheduler import WarmupCosineScheduler
from train_eval.common_training_setup import TrainingConfig, run_specific_experiment
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

    with open("data/some_text.txt", "r", encoding="utf-8") as file:
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

    start_context = "Hello, I am"
    print("Input text:", start_context)

    out = model.generate_text(contexts=start_context, max_length=30, temperature=1.2)
    print(out)

    print("Training Done!")


if __name__ == "__main__":
    model_config = load_config("configs/gpt_124M.yaml")
    training_config = load_config("configs/next_token_training.yaml")
    main(training_config, model_config)