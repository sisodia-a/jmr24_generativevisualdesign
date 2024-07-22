"""
Project: Generative Interpretable Visual Design: Using Disentanglement for Visual Conjoint Analysis
Author: Ankit Sisodia, Alex Burnap, Vineet Kumar
Email: asisodia@purdue.edu, alex.burnap@yale.edu, vineet.kumar@yale.edu
Date: July 2024
License: MIT
"""
import torch
from torch import nn

def get_activation_name(activation):
    """Given a string or a `torch.nn.modules.activation` return the name of the activation."""
    if isinstance(activation, str):
        return activation

    mapper = {nn.LeakyReLU: "leaky_relu", nn.ReLU: "relu", nn.Tanh: "tanh",
              nn.Sigmoid: "sigmoid", nn.Softmax: "sigmoid"}
    for k, v in mapper.items():
        if isinstance(activation, k):
            return k

    raise ValueError("Unkown given activation type : {}".format(activation))

def get_gain(activation):
    """Given an object of `torch.nn.modules.activation` or an activation name
    return the correct gain."""
    if activation is None:
        return 1

    activation_name = get_activation_name(activation)

    param = None if activation_name != "leaky_relu" else 0.2
    gain = nn.init.calculate_gain(activation_name, param)

    return gain

def linear_init(layer, activation="relu"):
    """Initialize a linear layer.
    Args:
        layer (nn.Linear): parameters to initialize.
        activation (`torch.nn.modules.activation` or str, optional) activation that
            will be used on the `layer`.
    """
    x = layer.weight

    if activation is None:
        return nn.init.xavier_uniform_(x)

    activation_name = get_activation_name(activation)

    if activation_name == "leaky_relu":
        a = 0 if isinstance(activation, str) else 0.2
        return nn.init.kaiming_uniform_(x, a=a, nonlinearity='leaky_relu')
    elif activation_name == "relu":
        return nn.init.kaiming_uniform_(x, nonlinearity='relu')
    elif activation_name in ["sigmoid", "tanh"]:
        return nn.init.xavier_uniform_(x, gain=get_gain(activation))
    elif activation_name in []:
        return nn.init.xavier_uniform_(x, gain=get_gain())

def weights_init(module):
    if isinstance(module, torch.nn.modules.conv._ConvNd):
        linear_init(module)
    elif isinstance(module, nn.Linear):
        linear_init(module)
