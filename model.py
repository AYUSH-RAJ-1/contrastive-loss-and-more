import torch
from torch import nn
import numpy as np
import random

class CNNEmbedding(nn.Module):
    

    def __init__(self):
        super().__init__()
        # input of n image  with 32*32 
        # their is also something called depthwise convolution + pointwise convolution for mobile net'self
        # one single 7*7 layer and 3 3*3 layer will have same receptive field and 3*3 will have less parameters
        # do above excersise
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # simple to calculate and easy gradient passing
            # increasing the receptive field of the information
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,

            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)

        )

        # we need flattened output
        self.flatten = nn.Flatten()# our dimention[0] will be preserved and after all that will be merged

        self.embedding = nn.Linear[64*7*7,128]# linear layer for create output latent vector

    def forward(self,x):
        x = self.sequential(x)
        x = self.flatten(x)
        return (self.embedding(x))

    
# input to this should be a torch

# we want our model to create a model use same weights and then predict two distances and compare loss
