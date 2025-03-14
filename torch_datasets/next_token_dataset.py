import torch
from torch.utils.data import Dataset
import random

class NextTokenDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride, random_sampling=False, num_samples=None, min_length=32):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.random_sampling = random_sampling
        self.min_length = min_length

        token_ids = tokenizer.encode(txt)
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.num_tokens = len(self.token_ids) - 1

        self.dataset_length = max((self.num_tokens - self.max_length) // self.stride, 0)

        remainder = (self.num_tokens - self.max_length) % self.stride
        if remainder >= self.min_length:
            self.dataset_length += 1  # Allow the last partial sequence

        if self.random_sampling:
            # this is mostly a test to be honest. I don't think it is good
            self.num_samples = num_samples if num_samples is not None else self.dataset_length * 2
        else:
            self.num_samples = self.dataset_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.random_sampling:
            start_idx = random.randint(0, self.num_tokens - self.min_length)
        else:
            start_idx = idx * self.stride

        end_idx = start_idx + self.max_length

        if end_idx > self.num_tokens:
            input_chunk = self.token_ids[start_idx:self.num_tokens]
            target_chunk = self.token_ids[start_idx + 1:self.num_tokens + 1]

            # Discard if too short
            if len(input_chunk) < self.min_length:
                raise IndexError("Sample too short. This should not happen during normal iteration.")

        else:
            input_chunk = self.token_ids[start_idx:end_idx]
            target_chunk = self.token_ids[start_idx + 1:end_idx + 1]

        return {
            "input_ids": input_chunk,
            "labels": target_chunk
        }
