import torch
import torch.nn as nn
from torch import optim

# Create a model
model = nn.Linear(10, 1)

# Create an optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward pass
    optimizer.zero_grad()  # Clear gradients from previous step
    loss.backward()        # Compute gradients
    optimizer.step()       # Update parameters


# Create optimizer with specific hyperparameters
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# You can also optimize specific parameter groups
optimizer = optim.Adam([
    {'params': model.conv_layers.parameters(), 'lr': 0.001},
    {'params': model.fc_layers.parameters(), 'lr': 0.01}
])

#basic SGD optimizer
'''
The optim.SGD() doesn't contain any stochasticity by itself.
The stochasticity comes from how you use it in your training loop, specifically how you handle your data.
'''
optimizer = optim.SGD(
    model.parameters(), 
    lr=0.001,           # Learning rate
    momentum=0.0,       # Momentum factor (0 = no momentum)
    weight_decay=0.0,   # L2 regularization
    dampening=0.0,      # Dampening for momentum
    nesterov=False      # Use Nesterov momentum
)


#manually with all data (no batching)
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Process entire dataset at once (not in batches)
for epoch in range(num_epochs):
    outputs = model(all_inputs)  # All data at once
    loss = criterion(outputs, all_targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()