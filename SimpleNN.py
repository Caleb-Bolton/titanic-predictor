import torch
import torch.nn as nn
import torch.nn.functional as F

# Define neural network
class SimpleNN(nn.Module):
    def __init__(self, layers, device=None):
        super(SimpleNN, self).__init__()
        self.layers = nn.ModuleList()
        self.device = device
        # Create the layers
        for i in range(1, len(layers)):
            self.layers.append(nn.Linear(layers[i-1], layers[i]))

    def forward(self, x):
        # Pass through each layer
        for i in range(len(self.layers)):
            layer = self.layers[i]
            # If it's the last layer, don't apply any activation function (make it linear)
            if i == len(self.layers) - 1:
                x = F.sigmoid(layer(x))
            else:
                x = F.relu(layer(x))
        return x