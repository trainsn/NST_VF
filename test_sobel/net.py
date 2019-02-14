import torch
from torch import nn

import numpy as np
import pdb

class Sobel(torch.nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        a = np.array([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]])
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1.weight = nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0))

        b = np.array([[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]])
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2.weight = nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0))

    def forward(self, X):
        G_x = self.conv1(X)
        G_y = self.conv2(X)
        G = torch.cat((-G_y[0], G_x[0]), 0).unsqueeze(0)
        return G

class CosineLoss(torch.nn.Module):
    def __init__(self, reduce=True):
        super(CosineLoss, self).__init__()
        self.reduce = reduce

    def forward(self, target, output):
        target_norm = target / torch.sqrt(torch.pow(target, 2).sum(1))
        output_norm = output / torch.sqrt(torch.pow(output, 2).sum(1))
        dot = 1 - target_norm.mul(output_norm).sum(1)
        if self.reduce:
            return dot.mean()
        else:
            return dot.sum()



