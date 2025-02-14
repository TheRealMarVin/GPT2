from helpers.config_helpers import load_config
from models.gpt import GPT


def main(config):
    model = GPT(config)

if __name__ == "__main__":
    config = load_config("configs/gpt_124M.yaml")
    main(config)