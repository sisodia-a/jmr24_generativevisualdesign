'''
        Project: Generative Interpretable Visual Design: Using Disentanglement for Visual Conjoint Analysis
        Authors: Ankit Sisodia, Purdue University
        Email: asisodia@purdue.edu

        Portions of Code From or Modified from Open Source Projects:
        https://github.com/YannDubs/disentangling-vae
'''
import argparse
import logging
import sys
import os
from configparser import ConfigParser

from torch import optim

from models.vae import VAE, Encoder, Decoder, Encoder_VGG, Decoder_VGG
from models.regression import WTPregression
from training.training import Trainer
from training.evaluate import Evaluator
from models.modelIO import save_model, load_model, load_metadata
from models.losses import BtcvaeLoss
from dataset.datasets import get_dataloaders, get_img_size
from utils.helpers import (create_safe_directory, get_device, set_seed, get_n_param,get_config_section)
from utils.visualize import Visualizer
from utils.viz_helpers import get_samples
from torchsummary import summary

CONFIG_FILE = "hyperparam.ini"
RES_DIR = "results"
LOG_LEVELS = list(logging._levelToName.values())
ADDITIONAL_EXP = ['custom']
EXPERIMENTS = ADDITIONAL_EXP

def parse_arguments(args_to_parse):
    """Parse the command line arguments.

    Parameters
    ----------
    args_to_parse: list of str
        Arguments to parse (splitted on whitespaces).
    """
    default_config = get_config_section([CONFIG_FILE], "Custom")
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str,help="Name of the model for storing and loading purposes.", default=default_config['name'])
    parser.add_argument('-s', '--seed', type=int, default=default_config['seed'],help='Random seed. Can be `None` for stochastic behavior.')
    parser.add_argument('-mt', '--model-type', type=str,default=default_config['model_type'],help='Type of Model')
    parser.add_argument('-tv', '--threshold-val', type=float,default=default_config['threshold_val'],help='Threshold for Masking.')
    parser.add_argument('-i', '--idcs', type=int, nargs='+', default=[],help='List of indices to of images to put at the begining of the samples.')
    args = parser.parse_args()
    return args


def main(args):
    """Main function for plotting fro pretrained models.

    Parameters
    ----------
    args: argparse.Namespace
        Arguments
    """
    default_config = get_config_section([CONFIG_FILE], "Custom")
    logger = logging.getLogger(__name__)

    set_seed(args.seed)
    experiment_name = args.name
    model_dir = os.path.join(RES_DIR, experiment_name)
    meta_data = load_metadata(model_dir)
    model = load_model(model_dir,model_type=args.model_type,threshold_val=args.threshold_val)
    model.eval()  # don't sample from latent: use mean
    dataset = "watches"
    train_loader, test1_loader, test2_loader, train_loader_unshuffled, train_loader_batch1 = get_dataloaders("watches",batch_size=int(default_config['batch_size']),eval_batchsize=int(default_config['eval_batchsize']),model_name=args.name,logger=logger)

    viz = Visualizer(model=model,
                     model_dir=model_dir,
                     experiment_name=args.name,
                     dataset=dataset,
                     max_traversal=int(default_config['max_traversal']),
                     loss_of_interest='kl_loss_training_',
                     upsample_factor=int(default_config['upsample_factor']))
    size = (int(default_config['n_rows']), int(default_config['n_cols']))
    # same samples for all plots: sample max then take first `x`data  for all plots
    num_samples = int(default_config['n_cols']) * int(default_config['n_rows'])
    samples = get_samples(train_loader_batch1, num_samples, idcs=default_config['idcs'])

    viz.generate_samples(size=size)
    viz.data_samples(samples)
    viz.traversals(data=samples[0:1, ...] if default_config['is_posterior'] else None,n_per_latent=int(default_config['n_cols'])*2,n_latents=int(default_config['n_rows']),is_reorder_latents=True)
    viz.reconstruct_traverse(samples,is_posterior=default_config['is_posterior'],n_latents=int(default_config['n_rows']),n_per_latent=int(default_config['n_cols'])*2,is_show_text=default_config['is_show_loss'])
    viz.save_cbc_images()

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
