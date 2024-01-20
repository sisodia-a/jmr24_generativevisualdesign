"""
Ankit's Module containing the regression model
"""
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

class WTPregression(nn.Module):
    def __init__(self,n_feature=10,n_output=1):
        super(WTPregression,self).__init__()
        self.predict=nn.Linear(n_feature,n_output)
    def forward(self,x):
        x = self.predict(x)
        return x
