# dataset_lab_cached.py
import os
import torch
from torch.utils.data import Dataset

class CachedLABDataset(Dataset):
    def __init__(self, cache_dir):
        self.files = [
            os.path.join(cache_dir, f)
            for f in os.listdir(cache_dir)
            if f.endswith(".pt")
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = torch.load(self.files[idx])
        return sample["L"], sample["ab"]
