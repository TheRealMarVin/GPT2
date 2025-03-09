import numpy as np
import torch
from torch.utils.data import Dataset


class AlpacaDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        merged_data = np.concatenate(self.data[idx])
        input = merged_data[:-1]
        output = merged_data[1:]
        content_tensor = torch.tensor(input, dtype=torch.long)
        label_tensor = torch.tensor(output, dtype=torch.long)
        return content_tensor, label_tensor
