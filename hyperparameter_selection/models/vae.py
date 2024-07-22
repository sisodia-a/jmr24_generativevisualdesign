"""
Project: Generative Interpretable Visual Design: Using Disentanglement for Visual Conjoint Analysis
Author: Ankit Sisodia, Alex Burnap, Vineet Kumar
Email: asisodia@purdue.edu, alex.burnap@yale.edu, vineet.kumar@yale.edu
Date: July 2024
License: MIT
Notes: Module containing the main VAE class.
"""
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from models.initialization import weights_init
from models.regression import WTPregression

class VAE(nn.Module):
    def __init__(self, img_size, encoder, decoder, regression, latent_dim, model_type, threshold_val,sup_signal):
        """
        Class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
        """
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_pixels = self.img_size[1] * self.img_size[2]
        self.encoder = encoder(self.img_size, self.latent_dim)
        self.model_type = model_type
        self.threshold_val = threshold_val
        self.sup_signal = sup_signal

        if self.sup_signal == 'brand':
            self.regression = regression(self.latent_dim,4)
        elif self.sup_signal == 'circa':
            self.regression = regression(self.latent_dim,7)
        elif self.sup_signal == 'material':
            self.regression = regression(self.latent_dim,3)
        elif self.sup_signal == 'movement':
            self.regression = regression(self.latent_dim,2)
        elif self.sup_signal == 'discreteprice':
            self.regression = regression(self.latent_dim,1)
        elif self.sup_signal == 'discreteprice_brand':
            self.regression = regression(self.latent_dim,9)
        elif self.sup_signal == 'discreteprice_circa':
            self.regression = regression(self.latent_dim,15)
        elif self.sup_signal == 'discreteprice_material':
            self.regression = regression(self.latent_dim,7)
        elif self.sup_signal == 'discreteprice_movement':
            self.regression = regression(self.latent_dim,5)
        elif self.sup_signal == 'brand_circa':
            self.regression = regression(self.latent_dim,38)
        elif self.sup_signal == 'brand_material':
            self.regression = regression(self.latent_dim,19)
        elif self.sup_signal == 'brand_movement':
            self.regression = regression(self.latent_dim,14)
        elif self.sup_signal == 'circa_material':
            self.regression = regression(self.latent_dim,31)
        elif self.sup_signal == 'circa_movement':
            self.regression = regression(self.latent_dim,22)
        elif self.sup_signal == 'material_movement':
            self.regression = regression(self.latent_dim,11)
        elif self.sup_signal == 'discreteprice_brand_circa':
            self.regression = regression(self.latent_dim,76)
        elif self.sup_signal == 'discreteprice_brand_material':
            self.regression = regression(self.latent_dim,39)
        elif self.sup_signal == 'discreteprice_brand_movement':
            self.regression = regression(self.latent_dim,29)
        elif self.sup_signal == 'discreteprice_circa_material':
            self.regression = regression(self.latent_dim,61)
        elif self.sup_signal == 'discreteprice_circa_movement':
            self.regression = regression(self.latent_dim,43)
        elif self.sup_signal == 'discreteprice_material_movement':
            self.regression = regression(self.latent_dim,23)
        elif self.sup_signal == 'brand_circa_material':
            self.regression = regression(self.latent_dim,133)
        elif self.sup_signal == 'brand_circa_movement':
            self.regression = regression(self.latent_dim,98)
        elif self.sup_signal == 'brand_material_movement':
            self.regression = regression(self.latent_dim,56)
        elif self.sup_signal == 'circa_material_movement':
            self.regression = regression(self.latent_dim,80)
        elif self.sup_signal == 'discreteprice_brand_circa_material':
            self.regression = regression(self.latent_dim,230)
        elif self.sup_signal == 'discreteprice_brand_circa_movement':
            self.regression = regression(self.latent_dim,182)
        elif self.sup_signal == 'discreteprice_brand_material_movement':
            self.regression = regression(self.latent_dim,102)
        elif self.sup_signal == 'discreteprice_circa_material_movement':
            self.regression = regression(self.latent_dim,147)
        elif self.sup_signal == 'brand_circa_material_movement':
            self.regression = regression(self.latent_dim,275)
        elif self.sup_signal == 'discreteprice_brand_circa_material_movement':
            self.regression = regression(self.latent_dim,437)
        elif self.sup_signal == 'continuousprice':
            self.regression = regression(self.latent_dim)

        self.decoder = decoder(self.img_size, self.latent_dim)

        self.reset_parameters()

    def reparameterize(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean

    def meaningful_visual_attributes(self, mean, logvar, threshold_val):
        """
        """
        latent_dim = mean.size(1)
        batch_size = mean.size(0)
        latent_kl = 0.5 * (-1 - logvar + mean.pow(2) + logvar.exp()).mean(dim=0)
        zeros = torch.zeros([mean.size(0),mean.size(1)])
        ones = torch.ones([mean.size(0),mean.size(1)])

        for i in range(latent_dim):
            if latent_kl[i].item() < threshold_val:
                ones[:,i] = zeros[:,i]

        return ones

    def forward(self, x, signal_value, continuousprice, discreteprice, brand, circa, material, movement, discreteprice_brand, discreteprice_circa, discreteprice_material, discreteprice_movement, brand_circa, brand_material, brand_movement, circa_material, circa_movement, material_movement, discreteprice_brand_circa, discreteprice_brand_material, discreteprice_brand_movement, discreteprice_circa_material, discreteprice_circa_movement, discreteprice_material_movement, brand_circa_material, brand_circa_movement, brand_material_movement, circa_material_movement, discreteprice_brand_circa_material, discreteprice_brand_circa_movement, discreteprice_brand_material_movement, discreteprice_circa_material_movement, brand_circa_material_movement, discreteprice_brand_circa_material_movement):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        continuousprice_s = continuousprice.shape[0]
        continuousprice = torch.reshape(continuousprice,(continuousprice_s,1))

        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        reconstruct = self.decoder(latent_sample)
        visual_attributes = self.meaningful_visual_attributes(*latent_dist, self.threshold_val)

        signal_pred = self.regression(torch.mul(latent_dist[0],visual_attributes.cuda()))

        return reconstruct, latent_dist, latent_sample, signal_pred, visual_attributes

    def reset_parameters(self):
        self.apply(weights_init)

    def sample_latent(self, x):
        """
        Returns a sample from the latent distribution.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        return latent_sample

class Encoder(nn.Module):
    def __init__(self,
                 img_size,
                 latent_dim=10):
        r"""Encoder of the model proposed in [1].

        Parameters
        ----------
        img_size : tuple of ints

        latent_dim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256*2 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

        """
        super(Encoder, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256*2
        self.latent_dim = latent_dim
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(n_chan, hid_channels, kernel_size, stride=2, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(hid_channels, hid_channels, kernel_size, stride=2, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(hid_channels, hid_channels, kernel_size, stride=2, padding=1, dilation=1)
        self.conv_64 = nn.Conv2d(hid_channels, hid_channels, kernel_size, stride=2, padding=1, dilation=1)
        self.conv_128=nn.Conv2d(hid_channels, hid_channels, kernel_size, stride=2, padding=1, dilation=1)

        # Fully connected layers
        self.lin1 = nn.Linear(np.product(self.reshape), hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

        # Fully connected layers for mean and variance
        self.mu_logvar_gen = nn.Linear(hidden_dim, self.latent_dim * 2)

    def forward(self, x):
        batch_size = x.size(0)

        # Convolutional layers with ReLu activations
        x = torch.nn.functional.leaky_relu(self.conv1(x))
        x = torch.nn.functional.leaky_relu(self.conv2(x))
        x = torch.nn.functional.leaky_relu(self.conv3(x))
        x = torch.nn.functional.leaky_relu(self.conv_64(x))
        x = torch.nn.functional.leaky_relu(self.conv_128(x))

        # Fully connected layers with ReLu activations
        x = x.view((batch_size, -1))
        x = torch.nn.functional.leaky_relu(self.lin1(x))
        x = torch.nn.functional.leaky_relu(self.lin2(x))

        # Fully connected layer for log variance and mean
        # Log std-dev in paper (bear in mind)
        mu_logvar = self.mu_logvar_gen(x)
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, img_size,
                 latent_dim=10):
        r"""Decoder of the model proposed in [1].

        Parameters
        ----------
        img_size : tuple of ints

        latent_dim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256*2 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)
        """
        super(Decoder, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256*2
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]
        self.img_size = img_size

        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, np.product(self.reshape))

        # Convolutional layers
        self.convT_128 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, stride=2, padding=1, dilation=1)
        self.convT_64 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, stride=2, padding=1, dilation=1)
        self.convT1 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, stride=2, padding=1, dilation=1)
        self.convT2 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, stride=2, padding=1, dilation=1)
        self.convT3 = nn.ConvTranspose2d(hid_channels, n_chan, kernel_size, stride=2, padding=1, dilation=1)

    def forward(self, z):
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = torch.nn.functional.leaky_relu(self.lin1(z))
        x = torch.nn.functional.leaky_relu(self.lin2(x))
        x = torch.nn.functional.leaky_relu(self.lin3(x))
        x = x.view(batch_size, *self.reshape)

        # Convolutional layers with ReLu activations
        x = torch.nn.functional.leaky_relu(self.convT_128(x))
        x = torch.nn.functional.leaky_relu(self.convT_64(x))
        x = torch.nn.functional.leaky_relu(self.convT1(x))
        x = torch.nn.functional.leaky_relu(self.convT2(x))
        # Sigmoid activation for final conv layer
        x = torch.sigmoid(self.convT3(x))

        return x

