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

from models.vae import VAE, Encoder, Decoder
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
    default_config = get_config_section([CONFIG_FILE], "Custom")
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str,help="Name of the model for storing and loading purposes.", default=default_config['name'])
    parser.add_argument('--btcvae-A', type=float,default=default_config['btcvae_A'],help="Weight of the MI term (alpha in the paper).")
    parser.add_argument('--btcvae-B', type=float,default=default_config['btcvae_B'],help="Weight of the TC term (beta in the paper).")
    parser.add_argument('--btcvae-G', type=float,default=default_config['btcvae_G'],help="Weight of the dim-wise KL term (gamma in the paper).")
    parser.add_argument('--btcvae-M', type=float,default=default_config['btcvae_M'],help="Weight of the MSE term (delta in the paper).")
    parser.add_argument('-s', '--seed', type=int, default=default_config['seed'],help='Random seed. Can be `None` for stochastic behavior.')
    parser.add_argument('-e', '--epochs', type=int,default=default_config['epochs'],help='Maximum number of epochs to run for.')
    parser.add_argument('-mt', '--model-type', type=str,default=default_config['model_type'],help='Type of Model')
    parser.add_argument('-a', '--reg-anneal', type=float,default=default_config['reg_anneal'],help="Number of annealing steps where gradually adding the regularisation. What is annealed is specific to each loss.")
    parser.add_argument('-tv', '--threshold-val', type=float,default=default_config['threshold_val'],help='Threshold for Masking.')
    parser.add_argument('--sup_signal', type=str,default=default_config['sup_signal'],help="Choice of Signal")
    args = parser.parse_args(args_to_parse)
    return args

def main(args):
    """Main train and evaluation function.

    Parameters
    ----------
    args: argparse.Namespace
    """
    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                                  "%H:%M:%S")
    logger = logging.getLogger(__name__)

    default_config = get_config_section([CONFIG_FILE], "Custom")

    logger.setLevel(default_config['log_level'].upper())
    stream = logging.StreamHandler()
    stream.setLevel(default_config['log_level'].upper())
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    set_seed(int(args.seed))
    device = get_device(is_gpu=not default_config['no_cuda'])
    exp_dir = os.path.join(RES_DIR, args.name)
    model_dir = exp_dir
    logger.info("Root directory for saving and loading experiments: {}".format(exp_dir))

    create_safe_directory(exp_dir, logger=logger)

    # PREPARES DATA (what is the purpose of each?)
    train_loader, validation_loader, test1_loader, test2_loader, train_loader_unshuffled, train_loader_batch1 = get_dataloaders("watches",batch_size=int(default_config['batch_size']),eval_batchsize=int(default_config['eval_batchsize']),model_name=args.name,sup_signal=args.sup_signal,logger=logger)

    logger.info("Train {} with {} samples".format("watches", len(train_loader.dataset)))

    # PREPARES MODEL
    img_size = get_img_size("watches")  # stores for metadata
    default_config['img_size'] = img_size
    model = VAE(img_size,Encoder,Decoder,WTPregression,int(default_config['latent_dim']),args.model_type,threshold_val=args.threshold_val,sup_signal=args.sup_signal)
    logger.info('Num parameters in model: {}'.format(get_n_param(model)))

    # TRAINS
    optimizer = optim.Adam(model.parameters(), lr=float(default_config['lr']),betas=(0.9,0.999))
    model = model.to(device)  # make sure trainer and viz on same device
    loss_f = BtcvaeLoss(rec_dist=default_config['rec_dist'],steps_anneal=float(args.reg_anneal),n_data=len(train_loader.dataset),alpha=float(args.btcvae_A),beta=float(args.btcvae_B),gamma=float(args.btcvae_G),delta=float(args.btcvae_M),sup_signal=args.sup_signal)

    trainer = Trainer(model, optimizer, loss_f, device=device, logger=logger, save_dir=exp_dir, experiment_name=args.name, model_type=args.model_type, is_progress_bar=not default_config['no_progress_bar']) 
    trainer(train_loader,validation_loader,train_loader_unshuffled,epochs=int(args.epochs),checkpoint_every=int(default_config['checkpoint_every']),)

    # SAVE MODEL AND EXPERIMENT INFORMATION
    save_model(trainer.model, exp_dir, metadata=default_config)

    model = load_model(exp_dir, is_gpu=not default_config['no_cuda'], filename="model.pt", model_type=args.model_type,sup_signal=args.sup_signal,threshold_val=args.threshold_val)
    metadata = load_metadata(exp_dir)
    loss_f = BtcvaeLoss(rec_dist=default_config['rec_dist'],steps_anneal=float(args.reg_anneal),n_data=len(test1_loader.dataset),alpha=float(args.btcvae_A),beta=float(args.btcvae_B),gamma=float(args.btcvae_G),delta=float(args.btcvae_M),sup_signal=args.sup_signal)
    evaluator = Evaluator(model, loss_f,device=device,logger=logger,save_dir=exp_dir,experiment_name=args.name,model_type=args.model_type,is_progress_bar=not default_config['no_progress_bar'],file_type="test1")
    evaluator(test1_loader) 

    loss_f = BtcvaeLoss(rec_dist=default_config['rec_dist'],steps_anneal=float(args.reg_anneal),n_data=len(test2_loader.dataset),alpha=float(args.btcvae_A),beta=float(args.btcvae_B),gamma=float(args.btcvae_G),delta=float(args.btcvae_M),sup_signal=args.sup_signal)
    evaluator = Evaluator(model, loss_f,device=device,logger=logger,save_dir=exp_dir,experiment_name=args.name,model_type=args.model_type,is_progress_bar=not default_config['no_progress_bar'],file_type="test2")
    evaluator(test2_loader)

    ## Visualizer
    meta_data = load_metadata(model_dir)
    model = load_model(model_dir, model_type=args.model_type,threshold_val=args.threshold_val,sup_signal=args.sup_signal)
    model.eval()  # don't sample from latent: use mean
    dataset = "watches"
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
    viz.traversals(data=samples[0:1, ...] if default_config['is_posterior'] else None,n_per_latent=int(default_config['n_cols'])*2,n_latents=int(default_config['n_rows']),is_reorder_latents=True)
    viz.reconstruct_traverse(samples,is_posterior=default_config['is_posterior'],n_latents=int(default_config['n_rows']),n_per_latent=int(default_config['n_cols'])*2,is_show_text=default_config['is_show_loss'])

    train_file = os.path.join(exp_dir,"christies_train.npz")
    test1_file = os.path.join(exp_dir,"christies_test1.npz")
    test2_file = os.path.join(exp_dir,"christies_test2.npz")
    validation_file = os.path.join(exp_dir,"christies_validation.npz")
    os.remove(train_file)
    os.remove(test1_file)
    os.remove(test2_file)
    os.remove(validation_file)
     
if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
