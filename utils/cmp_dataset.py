import torch
from torch.utils.data import Dataset
from tqdm import tqdm

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


# for label aggregation
class CMPSentenceDataset(Dataset):
    # def __init__(self, X, y, target, tokenizer):
    def __init__(self, X, y, target):
        self.X = X
        # self.X_input_ids, self.X_attention_mask = tokenize_all(X, tokenizer)

        if target in (RILE, GALTAN):
            self.y = torch.tensor(y[f'{target}_label_num'].to_list())
        elif target == JOINT:
            self.y = torch.tensor(y.to_numpy())
        self.y = self.y.long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # return {'input_ids': self.X_input_ids[idx],
        #             'attention_mask': self.X_attention_mask[idx],
        #             'labels': self.y[idx]}
        return {'texts': self.X[idx],
                'labels': self.y[idx]}


# for chunk-level regression
class CMPChunkDataset(Dataset):
    def __init__(self, X, y, target):
        self.X = X

        if target in (RILE, GALTAN):
            self.y = torch.tensor(y[f'chunk_{target}_score'].to_list())
        elif target == JOINT:
            self.y = torch.tensor(y.to_numpy())
        self.y = self.y.float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {'texts': self.X[idx],
                'scores': self.y[idx]}


def tokenize_all(X, tokenizer):
    n = len(X)
    input_ids = [None] * n
    attention_mask = [None] * n

    for i in tqdm(range(n), desc='Tokenizing'):
        tokens_i = tokenizer(X[i],
                             truncation=True,
                             max_length=100,
                             padding='max_length',
                             return_tensors="pt")
        input_ids[i] = tokens_i['input_ids'][0]
        attention_mask[i] = tokens_i['attention_mask'][0]

    return input_ids, attention_mask
