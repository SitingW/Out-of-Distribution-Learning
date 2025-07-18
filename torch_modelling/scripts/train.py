'''import libraries'''

# Add src directory to Python path for imports
# This runs automatically for all tests in this directory
import sys
import os
from pathlib import Path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))
import torch
import torch.nn as nn
from models.linear_model import LinearModel
from models.np_init_parameter import InitParameter
from data.dataset import LinearDataset
from data.data_generator import DataGenerator
from training.trainer import Trainer
from torch.utils.data import DataLoader
import numpy as np



'''set random seed for reproducibility'''
np.random.seed(42)
random_state = 42
'''defeine hyperparameters'''
learning_rate = 0.01
max_iterations = 1000
theta_0_num = 10

'''Generate sparse data'''
n_samples = 50
n_features = 100
output_features = 1
data_gen = DataGenerator(random_state = random_state)
X, y, _ = data_gen.get_linear_regression_data(n_samples=n_samples, n_features=n_features)
dataset = LinearDataset(X, y)


'''theta_0 generation'''
init_param = InitParameter(dim = n_features, n_samples = theta_0_num, random_state = random_state)
theta_0_array = init_param.initialization()
for j in range (theta_0_num):
    theta_0 = theta_0_array[:, j]
    '''modelling'''
    model = LinearModel(input_channels=n_features, output_channels=output_features, theta_0=theta_0)  # Example dimensions

trainer = Trainer(model, learning_rate=learning_rate)
#I should input dataset instead of X and y here
losses = trainer.train(dataset.X, dataset.y, epochs = max_iterations)
print(losses)