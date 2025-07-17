import torch
from torch.utils.data import Dataset
from .data_generator import DataGenerator



"""
I don't need to split data nor preprocess the data, as they are all generated.
but do I need a dataloader? what is the purpose of dataloader here? somewhat unnecessary?
"""
class SyntheticDataset(Dataset):
    def __init__(self, generator: DataGenerator, n_samples: int, 
                 n_features: int, transform=None):
        self.generator = generator
        self.n_samples = n_samples
        self.n_feature = n_features
        self.transform = transform
        
        # Generate all data at initialization or on-the-fly. With extra theta we will ignore it 
        self.X, self.y , _ = self.generator.get_linear_regression_data(n_samples, n_features)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx])
        y = torch.FloatTensor(self.y[idx])
        
        if self.transform:
            x = self.transform(x)
            
        return x, y