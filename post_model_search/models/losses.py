'''
Project: Generative Interpretable Visual Design: Using Disentanglement for Visual Conjoint Analysis
Author: Ankit Sisodia, Alex Burnap, Vineet Kumar
Email: asisodia@purdue.edu, alex.burnap@yale.edu, vineet.kumar@yale.edu
Date: July 2024
License: MIT

Notes: Portions of Code From or Modified from Open Source Projects:
       https://github.com/YannDubs/disentangling-vae
'''
import abc
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim

import logging

from models.math import (log_density_gaussian, log_importance_weight_matrix,
                               matrix_log_density_gaussian)

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class BtcvaeLoss():
    """
    Compute the decomposed KL loss with either minibatch weighted sampling or
    minibatch stratified sampling according to [1]

    Parameters
    ----------
    n_data: int
        Number of data in the training set

    alpha : float
        Weight of the mutual information term.

    beta : float
        Weight of the total correlation term.

    gamma : float
        Weight of the dimension-wise KL term.

    delta : float
        Weight of the MSE term

    is_mss : bool
        Whether to use minibatch stratified sampling instead of minibatch
        weighted sampling.

    kwargs:

    References
    ----------
       [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in variational
       autoencoders." Advances in Neural Information Processing Systems. 2018.
    """

    def __init__(self, record_loss_every=50, rec_dist="laplace", steps_anneal=0,n_data=1, alpha=1., beta=6., gamma=1., delta=1000.,sup_signal="brand",is_mss=True, **kwargs):
        super().__init__(**kwargs)

        self.n_train_steps = 0
        self.record_loss_every = record_loss_every
        self.rec_dist = rec_dist
        self.steps_anneal = steps_anneal

        self.n_data = n_data
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.delta = delta
        self.sup_signal = sup_signal
        self.is_mss = is_mss  # minibatch stratified sampling

    def __call__(self, data, recon_batch, latent_dist, is_train, storer, signal_pred, signal_value,
                 latent_sample=None):
        storer = self._pre_call(is_train, storer)
        batch_size, latent_dim = latent_sample.shape

        rec_loss = _reconstruction_loss(data, recon_batch,
                                        storer=storer,
                                        distribution=self.rec_dist)
        log_pz, log_qz, log_prod_qzi, log_q_zCx = _get_log_pz_qz_prodzi_qzCx(latent_sample,
                                                                             latent_dist,
                                                                             self.n_data,
                                                                             is_mss=self.is_mss)
        # I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
        mi_loss = (log_q_zCx - log_qz).mean()
        # TC[z] = KL[q(z)||\prod_i z_i]
        tc_loss = (log_qz - log_prod_qzi).mean()
        # dw_kl_loss is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
        dw_kl_loss = (log_prod_qzi - log_pz).mean()

        anneal_reg = (linear_annealing(0, 1, self.n_train_steps, self.steps_anneal)
                      if is_train else 1)
        x = list(signal_value.size())

        if self.sup_signal == 'continuousprice':
           signal_value = np.reshape(signal_value,(x[0],1)) ## PREDICTION_PROBLEM
           sup_loss = nn.MSELoss()(signal_pred.float().cuda(),signal_value.float().cuda())
        else:
           criterion = nn.CrossEntropyLoss() ## CLASSIFICATION_PROBLEM
           sup_loss = criterion(signal_pred.cuda(),signal_value.long().cuda()) 

        rsq = np.sum((signal_pred.cpu().detach().numpy()-np.sum(signal_value.cpu().detach().numpy())/len(signal_value.cpu().detach().numpy()))**2)/np.sum((signal_value.cpu().detach().numpy()-np.sum(signal_value.cpu().detach().numpy())/len(signal_value.cpu().detach().numpy()))**2)

        latent_kl = 0.5 * (-1 - latent_dist[1] + latent_dist[0].pow(2) + latent_dist[1].exp()).mean(dim=0)
        threshold = 0
        for i in range(latent_dim):
           if latent_kl[i].item() >= 0.75:
              threshold += 1

        loss =  rec_loss + (self.alpha * mi_loss + self.beta * tc_loss + self.gamma * dw_kl_loss) + anneal_reg * self.delta * sup_loss

        return loss, rec_loss, self.alpha * mi_loss, self.beta * tc_loss, self.gamma * dw_kl_loss, anneal_reg * self.delta * sup_loss, rsq, latent_kl

    def _pre_call(self, is_train, storer):
        if is_train:
            self.n_train_steps += 1

        if not is_train or self.n_train_steps % self.record_loss_every == 1:
            storer = storer
        else:
            storer = None
        return storer

def _reconstruction_loss(data, recon_data, distribution="laplace", storer=None):
    """
    Calculates the per image reconstruction loss for a batch of data. I.e. negative
    log likelihood.

    Parameters
    ----------
    data : torch.Tensor
        Input data (e.g. batch of images). Shape : (batch_size, n_chan,height, width).

    recon_data : torch.Tensor
        Reconstructed data. Shape : (batch_size, n_chan, height, width).

    distribution : {"bernoulli", "gaussian", "laplace"}
        Distribution of the likelihood on the each pixel. Implicitely defines the
        loss Bernoulli corresponds to a binary cross entropy (bse) loss and is the
        most commonly used. It has the issue that it doesn't penalize the same
        way (0.1,0.2) and (0.4,0.5), which might not be optimal. Gaussian
        distribution corresponds to MSE, and is sometimes used, but hard to train
        ecause it ends up focusing only a few pixels that are very wrong. Laplace
        distribution corresponds to L1 solves partially the issue of MSE.

    storer : dict
        Dictionary in which to store important variables for vizualisation.

    Returns
    -------
    loss : torch.Tensor
        Per image cross entropy (i.e. normalized per batch but not pixel and
        channel)
    """
    batch_size, n_chan, height, width = recon_data.size()
    is_colored = n_chan == 3

    if distribution == "bernoulli":
        loss = F.binary_cross_entropy(recon_data, data, reduction="sum")
    elif distribution == "gaussian":
        loss = F.mse_loss(recon_data, data, reduction="sum")
    elif distribution == "laplace":
        loss = F.l1_loss(recon_data, data, reduction="sum")
        loss = loss * (loss != 0)  # masking to avoid nan
    else:
        # assert distribution not in RECON_DIST
        raise ValueError("Unknown distribution: {}".format(distribution))

    loss = loss / batch_size

    return loss

"""
# def _kl_normal_loss(mean, logvar, storer=None):
    latent_dim = mean.size(1)
    # batch mean of kl for each latent dimension
    latent_kl = 0.5 * (-1 - logvar + mean.pow(2) + logvar.exp()).mean(dim=0)
    total_kl = latent_kl.sum()

    if storer is not None:
        storer['kl_loss'].append(total_kl.item())
        for i in range(latent_dim):
            storer['kl_loss_' + str(i)].append(latent_kl[i].item())

    return total_kl
"""

def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed

# Batch TC specific
def _get_log_pz_qz_prodzi_qzCx(latent_sample, latent_dist, n_data, is_mss=True):
    batch_size, hidden_dim = latent_sample.shape

    # calculate log q(z|x)
    log_q_zCx = log_density_gaussian(latent_sample, *latent_dist).sum(dim=1)

    # calculate log p(z)
    # mean and log var is 0
    zeros = torch.zeros_like(latent_sample)
    log_pz = log_density_gaussian(latent_sample, zeros, zeros).sum(1)

    mat_log_qz = matrix_log_density_gaussian(latent_sample, *latent_dist)

    log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)-math.log(batch_size * n_data)       
    log_prod_qzi = (torch.logsumexp(mat_log_qz, dim=1, keepdim=False)-math.log(batch_size * n_data)).sum(1) 

    # is_mss=False
    if is_mss:                                                                                                                
        # use stratification
        log_iw_mat = log_importance_weight_matrix(batch_size, n_data).to(latent_sample.device)                                
        log_qz = torch.logsumexp(log_iw_mat + mat_log_qz.sum(2), dim=1, keepdim=False)                                        
        log_prod_qzi = torch.logsumexp(log_iw_mat.view(batch_size,batch_size,1)+mat_log_qz, dim=1, keepdim=False).sum(1)      

    return log_pz, log_qz, log_prod_qzi, log_q_zCx
