import torch
from torch.utils.data import Dataset


class NextTokenDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride, random_offset=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.random_offset = random_offset

        token_ids = tokenizer.encode(txt)
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)

        self.dataset_length = (len(self.token_ids) - 1 - self.max_length) // self.stride + 1
        self.dataset_length = max(self.dataset_length, 0)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.max_length

        input_chunk = self.token_ids[start_idx:end_idx]
        target_chunk = self.token_ids[start_idx + 1:end_idx + 1]

        return {
            "input_ids": input_chunk,
            "labels": target_chunk
        }
