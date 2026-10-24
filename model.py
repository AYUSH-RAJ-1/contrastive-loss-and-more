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
        self.flatten = nn.Flatten(start_dim=1)# our dimention[0] will be preserved and after all that will be merged, as dimention 0 is the no of images in our batch we dont want to merge across the batches

        self.embedding = nn.Linear(64*7*7,128)# linear layer for create output latent vector

    def forward(self,x):
        x = self.conv_layers(x)
        # print("error is here",x.shape)
        x = self.flatten(x)
        # print("checking flatten",x.shape)
        return (self.embedding(x))

    
# input to this should be a torch

# we want our model to create a model use same weights and then predict two distances and compare loss

class SiameseNetwork(nn.Module):

    def __init__(self):

        super().__init__()
        self.embedding = CNNEmbedding()
    def forward(self, img1, img2):
        embd1 = self.embedding(img1)
        embd2 = self.embedding(img2)




        # calculate the distance which is l2 norm

        return embd1, embd2

def distance(embd1, embd2):
    return(
        torch.linalg.vector_norm( embd1 - embd2,ord=2,dim=1) # di
    )


if __name__ == "__main__":
    # testing the functions
    # create a test data of expected image dimentions pass it to the cnn embedding and then calculate if this is working
    img1 = torch.rand(4,1,28,28)
    img2 = torch.rand(4,1,28,28)
    # print(img1.shape)

    model = SiameseNetwork()

    embd1, embd2 = model(img1, img2)

    # print("embedding dimention",embd1.shape, embd2.shape)

    distance = distance(embd1,embd2)
    # print(distance)

