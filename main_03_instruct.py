import argparse
import json

from main_01_training_pre_train import train_next_token
from helpers.config_helpers import load_config
from sklearn.model_selection import train_test_split
from utils.instruction_helpers import format_input_for_alpaca, format_output_for_alpaca

def create_alpaca_datasets(file_path, model):
    tokenizer = model.tokenizer

    with open(file_path, "r") as file:
        data = json.load(file)

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

if __name__ == "__main__":
    default_model_config = "configs/gpt_124M_pre_trained.yaml"
    default_training_config = "configs/fine_tuning_training.yaml"

    parser = argparse.ArgumentParser(description="Configuration to launch training of the next token.")
    parser.add_argument("--model", type=str, default=default_model_config, help="Configuration to define the model to use during training.")
    parser.add_argument("--training", type=str, default=default_training_config, help="Configuration of the training routine.")
    args = parser.parse_args()

    model_config = load_config(args.model)
    training_config = load_config(args.training)
    train_next_token(training_config, model_config)