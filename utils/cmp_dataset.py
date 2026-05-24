import torch
from torch.utils.data import Dataset

import sys

sys.path.append('..')  # allows imports from parent / sibling directory
from utils.utils import RILE, GALTAN, JOINT, CLF, RGR


class CMPDataset(Dataset):
    def __init__(self, X, y, task, target):
        self.X = X
        self.y = y

        self.task = task
        self.target = target
        self._preprocess_y()

    def _preprocess_y(self):
        if self.target in (RILE, GALTAN):
            col = f'{self.target}_label_num' if self.task == CLF \
                else f'chunk_{self.target}_score'
            self.y = torch.tensor(self.y[col].to_list())
        elif self.target == JOINT:
            self.y = torch.tensor(self.y.to_numpy())

        if self.task == CLF:
            self.y = self.y.long()
        elif self.task == RGR:
            self.y = self.y.float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {'texts': self.X[idx],
                'ground_truth': self.y[idx]}
