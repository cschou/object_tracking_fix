import torch
import torch.nn as nn
import numpy as np

class EuclideanDist(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.sum(torch.sqrt(torch.sum(torch.pow((x-y), 2), axis=-1)))
    

if __name__ == '__main__':
    x = torch.tensor([[[4., 4.], [1., 1.]]])
    y = torch.tensor([[[1., 2.], [0., 0.]]])

    loss_fn = EuclideanDist()
    loss = loss_fn(x, y)
    print(loss)
    print(loss.shape)


    
