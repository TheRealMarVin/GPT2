import argparse
import os

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from main_01_training_pre_train import train_next_token
from torch_datasets.next_token_dataset import NextTokenDataset
from helpers.config_helpers import load_config
from models.gpt import GPT
from schedulers.warmup_cosine_scheduler import WarmupCosineScheduler
from train_eval.common_training_setup import TrainingConfig, run_specific_experiment
from utils.download_sherlock_datasets import download_sherlock_dataset
from utils.next_token_training import next_token_train_epoch, next_token_evaluate


if __name__ == "__main__":
    default_model_config = "configs/gpt_124M_pre_trained.yaml"
    default_training_config = "configs/instruct_training_lora.yaml"

    parser = argparse.ArgumentParser(description="Configuration to launch training the complete model.")
    parser.add_argument("--model", type=str, default=default_model_config, help="Configuration to define the model to use during training.")
    parser.add_argument("--training", type=str, default=default_training_config, help="Configuration of the training routine.")
    args = parser.parse_args()

    model_config = load_config(args.model)
    training_config = load_config(args.training)

    train_next_token(training_config, model_config, post_fix="_lora")