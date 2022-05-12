import os
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

PROJECT_NAME = 'bees'
PROJECT_DIR = os.environ.get(f'{PROJECT_NAME.upper()}DIR', '/kaggle/working/')
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)
PROJECT_DIR = Path(PROJECT_DIR)


class MLPDataset(Dataset):
    def __init__(self, individuals, targets, transform=None):
        self.features = individuals.loc[:,
                        ~individuals.columns.isin(['date', 'bee_id'])].values
        self.targets = targets.values
        self.transform = transform['individuals']

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx):
        x = self.features[[idx]]
        x_transformed = self.transform.transform(x).squeeze()
        y = self.targets[idx]
        return torch.tensor(x_transformed, dtype=torch.float), torch.tensor(y,
                                                                            dtype=torch.float)
