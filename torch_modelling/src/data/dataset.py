"""
I don't need to split data nor preprocess the data, as they are all generated.
but do I need a dataloader? what is the purpose of dataloader here? somewhat unnecessary?
"""

import torch
from torch.utils.data import Dataset
from .data_generator import DataGenerator
import numpy as np


class LinearDataset(Dataset):
    def __init__(self, X, y, transform=None):
        """
        Args:
            X: numpy array of features
            y: numpy array of targets
            transform: optional transform to apply to features
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        features = self.X[idx]
        target = self.y[idx]
        
        if self.transform:
            features = self.transform(features)
            
        return features, target