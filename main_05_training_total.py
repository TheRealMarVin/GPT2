import argparse
import os

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from main_01_training_pre_train import train_next_token
from main_03_training_instruct import train_instruct
from torch_datasets.next_token_dataset import NextTokenDataset
from helpers.config_helpers import load_config
from models.gpt import GPT
from schedulers.warmup_cosine_scheduler import WarmupCosineScheduler
from train_eval.common_training_setup import TrainingConfig, run_specific_experiment
from utils.download_sherlock_datasets import download_sherlock_dataset
from utils.next_token_training import next_token_train_epoch, next_token_evaluate


if __name__ == "__main__":
    default_model_config = "configs/gpt_355M_pre_trained.yaml"
    default_fine_tuning_training_config = "configs/fine_tuning_training.yaml"
    default_instruct_training_config = "configs/instruct_training.yaml"

    parser = argparse.ArgumentParser(description="Configuration to launch training the complete model.")
    parser.add_argument("--model", type=str, default=default_model_config,
                        help="Configuration to define the model to use during training.")
    parser.add_argument("--fine_tune_training", type=str, default=default_fine_tuning_training_config,
                        help="Configuration of the training for fine tuning routine.")
    parser.add_argument("--instruct_training", type=str, default=default_instruct_training_config,
                        help="Configuration of the training for instruct routine.")
    args = parser.parse_args()

    model_config = load_config(args.model)
    fine_tuning_training_config = load_config(args.fine_tune_training)
    instruct_training_config = load_config(args.instruct_training)

    experiments = [("top_k", 3, 1.0), ("top_k", 3, 0.25), ("top_k", 3, 2), ("top_k", 1, 1.0), ("top_k", 40, 1.0),
                   ("top_p", 0.1, 1.0), ("top_p", 0.1, 0.5), ("top_p", 0.1, 1.5), ("top_p", 0.05, 1.0),
                   ("top_p", 1.0, 1.0)]
    test_data = [("What is the color of the ocean?", []), ("Describe a sunny day.", []),
                 ("Where do Sherlock Holmes and Dr. Watson live?", experiments)]
    with open("test_data/sherlock_questions.txt", 'r', encoding='utf-8') as file:
        lines = file.readlines()
        new_tests = [(x.strip(), []) for x in lines]
        test_data.extend(new_tests)

    model = train_next_token(fine_tuning_training_config, model_config, post_fix="_final")
    train_instruct(instruct_training_config, model_config, model=model, post_fix="_instruct_lora", test_data=test_data)