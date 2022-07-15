import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class OnlineShopping(Dataset):
    """online_shopping_10_cats数据集"""

    def __init__(self, text: list, labels: list):
        self.text = text
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        text = torch.LongTensor(self.text[idx])
        return text, label


class JenaClimate(Dataset):
    """jena_climate_2009_2016数据集"""

    def __init__(self, known_data: list, pred_temper: list):
        self.known_data = known_data
        self.pred_temper = pred_temper

    def __len__(self):
        return len(self.pred_temper)

    def __getitem__(self, idx):
        known_data = self.known_data[idx]
        pred_temper = self.pred_temper[idx]
        return known_data, pred_temper
