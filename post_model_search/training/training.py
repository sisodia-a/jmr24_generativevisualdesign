import imageio
import logging
import os
from timeit import default_timer
from collections import defaultdict
import pandas
import csv
import numpy as np
import math
import scipy
from tqdm import trange
import torch
from torch.nn import functional as F

from models.modelIO import save_model

TRAIN_LOSSES_LOGFILE = "train_losses.csv"

class Trainer():
    """
    Class to handle training of model.

    Parameters
    ----------
    model: disvae.vae.VAE

    optimizer: torch.optim.Optimizer

    loss_f: disvae.models.BaseLoss
        Loss function.

    device: torch.device, optional
        Device on which to run the code.

    logger: logging.Logger, optional
        Logger.

    save_dir : str, optional
        Directory for saving logs.

    is_progress_bar: bool, optional
        Whether to use a progress bar for training.
    """

    def __init__(self, model, optimizer, loss_f,
                 device=torch.device("cpu"),
                 logger=logging.getLogger(__name__),
                 save_dir="results",
                 experiment_name="temp",
                 is_progress_bar=True,
                 model_type='m2'):

        self.device = device
        self.model = model.to(self.device)
        self.loss_f = loss_f
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        self.is_progress_bar = is_progress_bar
        self.logger = logger
        self.optimizer = optimizer
        self.model_type = model_type
        self.losses_logger = LossesLogger(os.path.join(self.save_dir, TRAIN_LOSSES_LOGFILE))
        self.logger.info("Training Device: {}".format(self.device))

    def __call__(self,train_loader,train_loader_unshuffled,epochs=10,checkpoint_every=10):
        """
        Trains the model.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        epochs: int, optional
            Number of epochs to train the model for.

        checkpoint_every: int, optional
            Save a checkpoint of the trained model every n epoch.
        """
        start = default_timer()
        self.model.train()
        for epoch in range(epochs):
            self.logger.info('Epoch: {}'.format(epoch + 1))
            self.model.train()
            storer = defaultdict(list)
            mean_epoch_loss, mean_rec_loss, mean_mi_loss, mean_tc_loss, mean_dw_kl_loss, mean_sup_loss, mean_rsq, mean_latent_kl = self._train_epoch(train_loader, storer, epoch)
            self.logger.info('Epoch: {} Average loss per image in Training {:.2f}'.format(epoch + 1,mean_epoch_loss))
            storer['loss_training'].append(mean_epoch_loss)
            storer['mi_loss_training'].append(mean_mi_loss)
            storer['tc_loss_training'].append(mean_tc_loss)
            storer['dw_kl_loss_training'].append(mean_dw_kl_loss)
            storer['sup_loss_training'].append(mean_sup_loss)
            storer['recon_loss_training'].append(mean_rec_loss)
            storer['rsq_training'].append(mean_rsq)
            for i in range(mean_latent_kl.shape[0]):
                storer['kl_loss_training_' + str(i)].append(mean_latent_kl[i].item())

            self.losses_logger.log(epoch, storer)

            if epoch % checkpoint_every == 0:
                save_model(self.model, self.save_dir,filename="model-{}.pt".format(epoch))

            is_still_training = False
            self.model.eval()

        self.model.eval()

        delta_time = (default_timer() - start) / 60
        self.logger.info('Finished training after {:.1f} min.'.format(delta_time))

        metrics = self.compute_metrics(train_loader_unshuffled,self.experiment_name+"_mean_params_train.csv",self.experiment_name+"_logvar_params_train.csv",self.experiment_name+"_filename_train.csv") ## CLASSIFICATION_PROBLEM

    def _train_epoch(self, data_loader, storer, epoch):
        """
        Trains the model for one epoch.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        storer: dict
            Dictionary in which to store important variables for vizualisation.

        epoch: int
            Epoch number

        Return
        ------
        mean_epoch_loss: float
            Mean loss per image
        """
        epoch_loss = 0.
        epoch_rec_loss = 0.
        epoch_mi_loss = 0.
        epoch_tc_loss = 0.
        epoch_dw_kl_loss = 0.
        epoch_sup_loss = 0.
        epoch_rsq = 0.
        epoch_latent_kl = 0.
        kwargs = dict(desc="Epoch {}".format(epoch + 1), leave=False,
                      disable=not self.is_progress_bar)
        with trange(len(data_loader), **kwargs) as t:
            for _, (data, _, signal_value, continuousprice, discreteprice, brand, circa, material, movement, discreteprice_brand, discreteprice_circa, discreteprice_material, discreteprice_movement, brand_circa, brand_material, brand_movement, circa_material, circa_movement, material_movement, discreteprice_brand_circa, discreteprice_brand_material, discreteprice_brand_movement, discreteprice_circa_material, discreteprice_circa_movement, discreteprice_material_movement, brand_circa_material, brand_circa_movement, brand_material_movement, circa_material_movement, discreteprice_brand_circa_material, discreteprice_brand_circa_movement, discreteprice_brand_material_movement, discreteprice_circa_material_movement, brand_circa_material_movement, discreteprice_brand_circa_material_movement, filenames) in enumerate(data_loader):
                iter_loss, rec_loss, mi_loss, tc_loss, dw_kl_loss, sup_loss, rsq, latent_kl = self._train_iteration(data, storer, signal_value, continuousprice, discreteprice, brand, circa, material, movement, discreteprice_brand, discreteprice_circa, discreteprice_material, discreteprice_movement, brand_circa, brand_material, brand_movement, circa_material, circa_movement, material_movement, discreteprice_brand_circa, discreteprice_brand_material, discreteprice_brand_movement, discreteprice_circa_material, discreteprice_circa_movement, discreteprice_material_movement, brand_circa_material, brand_circa_movement, brand_material_movement, circa_material_movement, discreteprice_brand_circa_material, discreteprice_brand_circa_movement, discreteprice_brand_material_movement, discreteprice_circa_material_movement, brand_circa_material_movement, discreteprice_brand_circa_material_movement, filenames, epoch+1)
                batch_size, channel, height, width = data.size()
                epoch_loss += iter_loss * batch_size
                epoch_rec_loss += rec_loss * batch_size
                epoch_mi_loss += mi_loss * batch_size
                epoch_tc_loss += tc_loss * batch_size
                epoch_dw_kl_loss += dw_kl_loss * batch_size
                epoch_sup_loss += sup_loss * batch_size
                epoch_rsq += rsq * batch_size
                epoch_latent_kl += latent_kl * batch_size

                t.set_postfix(loss=iter_loss)
                t.set_postfix(loss=rec_loss)
                t.set_postfix(loss=mi_loss)
                t.set_postfix(loss=tc_loss)
                t.set_postfix(loss=dw_kl_loss)
                t.set_postfix(loss=sup_loss)
                t.set_postfix(loss=rsq)
                t.update()

        mean_epoch_loss = epoch_loss / len(data_loader.dataset)
        mean_rec_loss = epoch_rec_loss / len(data_loader.dataset)
        mean_mi_loss = epoch_mi_loss / len(data_loader.dataset)
        mean_tc_loss = epoch_tc_loss / len(data_loader.dataset)
        mean_dw_kl_loss = epoch_dw_kl_loss / len(data_loader.dataset)
        mean_sup_loss = epoch_sup_loss / len(data_loader.dataset)
        mean_rsq = epoch_rsq / len(data_loader.dataset)
        mean_latent_kl = epoch_latent_kl / len(data_loader.dataset)

        return mean_epoch_loss, mean_rec_loss, mean_mi_loss, mean_tc_loss, mean_dw_kl_loss, mean_sup_loss, mean_rsq, mean_latent_kl

    def _train_iteration(self, data, storer, signal_value, continuousprice, discreteprice, brand, circa, material, movement, discreteprice_brand, discreteprice_circa, discreteprice_material, discreteprice_movement, brand_circa, brand_material, brand_movement, circa_material, circa_movement, material_movement, discreteprice_brand_circa, discreteprice_brand_material, discreteprice_brand_movement, discreteprice_circa_material, discreteprice_circa_movement, discreteprice_material_movement, brand_circa_material, brand_circa_movement, brand_material_movement, circa_material_movement, discreteprice_brand_circa_material, discreteprice_brand_circa_movement, discreteprice_brand_material_movement, discreteprice_circa_material_movement, brand_circa_material_movement, discreteprice_brand_circa_material_movement, filenames, epoch):
        """
        Trains the model for one iteration on a batch of data.

        Parameters
        ----------
        data: torch.Tensor
            A batch of data. Shape : (batch_size, channel, height, width).

        storer: dict
            Dictionary in which to store important variables for vizualisation.
        """
        batch_size, channel, height, width = data.size()
        data = data.to(self.device)

        recon_batch, latent_dist, latent_sample, signal_pred, visual_attributes = self.model(data, signal_value, continuousprice, discreteprice, brand, circa, material, movement, discreteprice_brand, discreteprice_circa, discreteprice_material, discreteprice_movement, brand_circa, brand_material, brand_movement, circa_material, circa_movement, material_movement, discreteprice_brand_circa, discreteprice_brand_material, discreteprice_brand_movement, discreteprice_circa_material, discreteprice_circa_movement, discreteprice_material_movement, brand_circa_material, brand_circa_movement, brand_material_movement, circa_material_movement, discreteprice_brand_circa_material, discreteprice_brand_circa_movement, discreteprice_brand_material_movement, discreteprice_circa_material_movement, brand_circa_material_movement, discreteprice_brand_circa_material_movement)
        loss, rec_loss, mi_loss, tc_loss, dw_kl_loss, sup_loss, rsq, latent_kl = self.loss_f(data, recon_batch, latent_dist, self.model.training, storer, signal_pred, signal_value, latent_sample=latent_sample)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), rec_loss.item(), mi_loss.item(), tc_loss.item(), dw_kl_loss.item(), sup_loss.item(), rsq.item(), latent_kl

    def compute_metrics(self, dataloader,mean_name,logvar_name,file_name):
        """
        """
        mean_params, logvar_params, filenames = self._compute_q_zCx(dataloader)
        mean_params = [item for sublist in mean_params for item in sublist]
        logvar_params = [item for sublist in logvar_params for item in sublist]
        filenames = [item for sublist in filenames for item in sublist] 
        np.savetxt(os.path.join(self.save_dir,mean_name),mean_params,delimiter=",",fmt='%s')
        np.savetxt(os.path.join(self.save_dir,file_name),filenames,delimiter=",",fmt='%s')
        return 0

    def _compute_q_zCx(self, dataloader):
        """
        """
        indices = []
        mean = []
        logvar = []
        with torch.no_grad():
            for _,(data, label, signal_value, continuousprice, discreteprice, brand, circa, material, movement, discreteprice_brand, discreteprice_circa, discreteprice_material, discreteprice_movement, brand_circa, brand_material, brand_movement, circa_material, circa_movement, material_movement, discreteprice_brand_circa, discreteprice_brand_material, discreteprice_brand_movement, discreteprice_circa_material, discreteprice_circa_movement, discreteprice_material_movement, brand_circa_material, brand_circa_movement, brand_material_movement, circa_material_movement, discreteprice_brand_circa_material, discreteprice_brand_circa_movement, discreteprice_brand_material_movement, discreteprice_circa_material_movement, brand_circa_material_movement, discreteprice_brand_circa_material_movement, filenames) in enumerate(dataloader):
                data = data.to(self.device)
                recon_batch, latent_dist, latent_sample, signal_pred, visual_attributes = self.model(data, signal_value, continuousprice, discreteprice, brand, circa, material, movement, discreteprice_brand, discreteprice_circa, discreteprice_material, discreteprice_movement, brand_circa, brand_material, brand_movement, circa_material, circa_movement, material_movement, discreteprice_brand_circa, discreteprice_brand_material, discreteprice_brand_movement, discreteprice_circa_material, discreteprice_circa_movement, discreteprice_material_movement, brand_circa_material, brand_circa_movement, brand_material_movement, circa_material_movement, discreteprice_brand_circa_material, discreteprice_brand_circa_movement, discreteprice_brand_material_movement, discreteprice_circa_material_movement, brand_circa_material_movement, discreteprice_brand_circa_material_movement)
                batch_size, channel, height, width = data.size()
                mean_val, logvar_val = latent_dist
                dim = mean_val.size(1)
                mean_val = torch.mul(mean_val,visual_attributes.cuda())
                logvar_val = torch.mul(logvar_val,visual_attributes.cuda())

                mean.append(mean_val.cpu().detach().numpy())
                logvar.append(logvar_val.cpu().detach().numpy())
                indices.append(list(filenames))
        return mean, logvar, indices

    def compute_losses(self, dataloader, epoch):
        """Compute all test losses.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
        """
        storer = defaultdict(list)

        epoch_loss = 0.
        epoch_rec_loss = 0.
        epoch_mi_loss = 0.
        epoch_tc_loss = 0.
        epoch_dw_kl_loss = 0.
        epoch_sup_loss = 0.
        epoch_rsq = 0.
        epoch_latent_kl = 0.

        with trange(len(dataloader)) as t:
            for _, (data, _, signal_value, continuousprice, discreteprice, brand, circa, material, movement, discreteprice_brand, discreteprice_circa, discreteprice_material, discreteprice_movement, brand_circa, brand_material, brand_movement, circa_material, circa_movement, material_movement, discreteprice_brand_circa, discreteprice_brand_material, discreteprice_brand_movement, discreteprice_circa_material, discreteprice_circa_movement, discreteprice_material_movement, brand_circa_material, brand_circa_movement, brand_material_movement, circa_material_movement, discreteprice_brand_circa_material, discreteprice_brand_circa_movement, discreteprice_brand_material_movement, discreteprice_circa_material_movement, brand_circa_material_movement, discreteprice_brand_circa_material_movement, filenames) in enumerate(dataloader):
                data = data.to(self.device)
                recon_batch, latent_dist, latent_sample, signal_pred, visual_attributes = self.model(data, signal_value, continuousprice, discreteprice, brand, circa, material, movement, discreteprice_brand, discreteprice_circa, discreteprice_material, discreteprice_movement, brand_circa, brand_material, brand_movement, circa_material, circa_movement, material_movement, discreteprice_brand_circa, discreteprice_brand_material, discreteprice_brand_movement, discreteprice_circa_material, discreteprice_circa_movement, discreteprice_material_movement, brand_circa_material, brand_circa_movement, brand_material_movement, circa_material_movement, discreteprice_brand_circa_material, discreteprice_brand_circa_movement, discreteprice_brand_material_movement, discreteprice_circa_material_movement, brand_circa_material_movement, discreteprice_brand_circa_material_movement)
                loss, rec_loss, mi_loss, tc_loss, dw_kl_loss, sup_loss, rsq, latent_kl = self.loss_f(data, recon_batch, latent_dist, self.model.training,storer,signal_pred, signal_value, latent_sample=latent_sample)
                batch_size, channel, height, width = data.size()
                epoch_loss += loss.item() * batch_size
                epoch_rec_loss += rec_loss.item() * batch_size
                epoch_mi_loss += mi_loss.item() * batch_size
                epoch_tc_loss += tc_loss.item() * batch_size
                epoch_dw_kl_loss += dw_kl_loss.item() * batch_size
                epoch_sup_loss += sup_loss.item() * batch_size
                epoch_rsq += rsq.item() * batch_size
                epoch_latent_kl += latent_kl * batch_size
 
                t.set_postfix(loss=loss)
                t.set_postfix(loss=rec_loss)
                t.set_postfix(loss=mi_loss)
                t.set_postfix(loss=tc_loss)
                t.set_postfix(loss=dw_kl_loss)
                t.set_postfix(loss=sup_loss)
                t.set_postfix(loss=rsq)
                t.set_postfix(loss=latent_kl)
                t.update()

        mean_epoch_loss = epoch_loss / len(dataloader.dataset)
        mean_rec_loss = epoch_rec_loss / len(dataloader.dataset)
        mean_mi_loss = epoch_mi_loss / len(dataloader.dataset)
        mean_tc_loss = epoch_tc_loss / len(dataloader.dataset)
        mean_dw_kl_loss = epoch_dw_kl_loss / len(dataloader.dataset)
        mean_sup_loss = epoch_sup_loss / len(dataloader.dataset)
        mean_rsq = epoch_rsq / len(dataloader.dataset)
        mean_latent_kl = epoch_latent_kl / len(dataloader.dataset)

        storer['loss_validation'].append(mean_epoch_loss)
        storer['mi_loss_validation'].append(mean_mi_loss)
        storer['tc_loss_validation'].append(mean_tc_loss)
        storer['dw_kl_loss_validation'].append(mean_dw_kl_loss)
        storer['sup_loss_validation'].append(mean_sup_loss)
        storer['recon_loss_validation'].append(mean_rec_loss)
        storer['rsq_validation'].append(mean_rsq)
        for i in range(mean_latent_kl.shape[0]):
            storer['kl_loss_validation_' + str(i)].append(mean_latent_kl[i].item())

        self.losses_logger.log(epoch, storer)

        losses = {k: sum(v) for k, v in storer.items()}
        return losses

class LossesLogger(object):
    """Class definition for objects to write data to log files in a
    form which is then easy to be plotted.
    """

    def __init__(self, file_path_name):
        """ Create a logger to store information for plotting. """
        if os.path.isfile(file_path_name):
            os.remove(file_path_name)

        self.logger = logging.getLogger("losses_logger")
        self.logger.setLevel(1)  # always store
        file_handler = logging.FileHandler(file_path_name)
        file_handler.setLevel(1)
        self.logger.addHandler(file_handler)

        header = ",".join(["Epoch", "Loss", "Value"])
        self.logger.debug(header)

    def log(self, epoch, losses_storer):
        """Write to the log file """
        for k, v in losses_storer.items():
            log_string = ",".join(str(item) for item in [epoch, k, mean(v)])
            self.logger.debug(log_string)

# HELPERS
def mean(l):
    """Compute the mean of a list"""
    return sum(l) / len(l)
