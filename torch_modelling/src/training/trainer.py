import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging

#import torch.optim as optim
#from torch.utils.data import DataLoader, TensorDataset
import numpy as np



'''
In this trainer, we will not use ADAM optimiser for an experimental purpose, as we want to compare the model performance with the closed-form solution.
We will also not use stochastic gradient descent, and we will use the full dataset for each iteration.
'''
class Trainer:
    def __init__(self, model, learning_rate):
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()
        # Use SGD optimizer instead of manual updates
        self.optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    def train(self, X_tensor, y_tensor, epochs):
        '''
        X: pytorch tensors of feature
        Y: pytorch tensors of targets
        '''

        losses = [] #why I need to keep track of this?
        #parameter_history = [] #we won't keep track of parameter history and if we do, we will save checkpoint

        for epoch in range (epochs):
            predictions = self.model(X_tensor)
            loss = self.loss_fn(predictions.squeeze(), y_tensor)
             
            #backward pass
            self.optimizer.zero_grad()  #clear gradients
            loss.backward()             #compute gradient
            self.optimizer.step()       #update parameters

            losses.append(loss.item())

            #epoch size is small so we don't print for now

        return losses