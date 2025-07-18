import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, theta_0, bias = False):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.linear_layer = nn.Linear(input_channels, output_channels, bias=bias)

        #check if I have no biased term
        print("Has bias:", self.linear_layer.bias is not None)  # False
        print("Bias:", self.linear_layer.bias) 

        theta_0 = torch.tensor(theta_0, dtype=torch.float32)
        self.linear_layer.weight.data = theta_0   
    
    def forward(self, x) -> torch.Tensor:
       return self.linear_layer(x)