import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
from typing import Dict, Any, Optional, List, Union


#import torch.optim as optim
#from torch.utils.data import DataLoader, TensorDataset
import numpy as np




'''
In this trainer, we will not use ADAM optimiser for an experimental purpose, as we want to compare the model performance with the closed-form solution.
We will also not use stochastic gradient descent, and we will use the full dataset for each iteration.
'''
class Trainer:
    def __init__(self, config: Dict[str, any]):
        self.model = config.get("model")
        self.learning_rate = config.get("learning_rate")
        self.lambda_val = config.get("lambda_val", 0)
        self.lr = config.get('lr', 0.01)
        self.loss_fn = config.get( "loss_fn", nn.MSELoss(reduction = 'mean')) #aligned with our numpy models
        # Use SGD optimizer instead of manual updates
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

        # Store initial parameters
        self.initial_params = {name: param.clone().detach() 
                          for name, param in self.model.named_parameters()}
        
        self.parameter_history = []
        
    def train(self, X_tensor, y_tensor, epochs):
        '''
        X: pytorch tensors of feature
        Y: pytorch tensors of targets
        '''

        #losses = [] #why I need to keep track of this?
        #parameter_history = [] #we won't keep track of parameter history and if we do, we will save checkpoint

        for epoch in range (epochs):
            predictions = self.model(X_tensor)
            loss = self.loss_fn(predictions.squeeze(), y_tensor)

            # Add L2 penalty based on distance from initial parameters
            l2_penalty = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    l2_penalty += torch.norm(param - self.initial_params[name]) ** 2

            # Add penalty to loss
            total_loss = loss + self.lambda_val * l2_penalty


            #backward pass
            self.optimizer.zero_grad()  #clear gradients
            total_loss.backward()             #compute gradient
            self.optimizer.step()       #update parameters

            #losses.append(loss.item())
            self.parameter_history.append(self.model.state_dict())
            #epoch size is small so we don't print for now

    
    
    '''getting single prediction from cache'''
    def get_prediction(self, x_new, epoch):
        
        # Load epoch params
        for name, param in self.model.named_parameters():
            param.data = self.parameter_history[epoch][name]
            #it temporary changed the parameter in the models into the given epoch
        
        # Predict and cache
        with torch.no_grad():
            pred = self.model(x_new)
        
        # # Restore params (do I need to?)
        # for name, param in self.model.named_parameters():
        #     param.data = current_params[name]

        #deep copy
        return pred.clone()
    
    def iterative_mean(self, x_new, epoches, alpha_val):
        if not isinstance(x_new, torch.Tensor):
            x_new = torch.tensor(x_new, dtype=torch.float32)
        
        #compute f_bar initial
        for name, param in self.model.named_parameters():
            param.data = self.initial_params[name]
            with torch.no_grad():
                f_bar = self.model(x_new)
        #never restored the parameter yet        

        for epoch in range(epoches):
            f_bar = alpha_val * self.get_prediction(x_new, epoch) + (1-alpha_val) *f_bar
        return f_bar

