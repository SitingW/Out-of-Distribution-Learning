import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.linear_layer = nn.Linear(input_channels, output_channels, bias=False)

        #check if I have no biased term
        print("Has bias:", self.linear_layer.bias is not None)  # False
        print("Bias:", self.linear_layer.bias)    
    
    def forward(self, x) -> torch.Tensor:
       return self.linear_layer(x)