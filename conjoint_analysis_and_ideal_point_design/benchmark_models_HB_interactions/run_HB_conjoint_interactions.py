"""
#############################################################################################
    Project: Generative Interpretable Visual Design: Using Disentanglement for Visual Conjoint Analysis
    Author: Ankit Sisodia, Alex Burnap, Vineet Kumar
    Email: asisodia@purdue.edu, alex.burnap@yale.edu, vineet.kumar@yale.edu
    Date: July 2024
    License: MIT
#############################################################################################
"""
# Standard Libs
import os
import math
import glob

# Numerical and Plotting Libs
import numpy as np
import pandas as pd
# from PIL import Image
import statsmodels as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Bayesian Libs
import pymc as pm
import arviz as az
# import graphviz
# import pytensor as pt
# import aesara.tensor as at
import jax
import pymc.sampling_jax as sampling_jax

# -----------------------------------------------------------------------------------------------
#                                 Globals
# -----------------------------------------------------------------------------------------------

NUM_RESPONDENTS = 253
NUM_ALTERNATIVES = 2
NUM_ATTRIBUTES = 6
NUM_TASKS = 15
NUM_ATTRIBUTES_PLUS_INTERACTIONS = 6 + 15 # attributes + pairwise interactions
NUM_TRAIN_TASKS = 13
NUM_TEST_TASKS = 2
NUM_COVARIATES = 6

choice_and_demo_coords_main_effects = {"visual_attributes": ["dialcolor",
                                                "dialshape",
                                                "strapcolor",
                                                "dialsize",
                                                "knobsize",
                                                "rimcolor"],

                                      "covariates": ["DemoGender_male",
                                                     "DemoGender_female",
                                                     "DemoAge_real",
                                                     "DemoIncome_real",
                                                     "DemoEducation_real",
                                                     "DemoAestheticImportance1_1_real",
                                                     # "DemoTimeSpentGrooming_real" # left out
                                                     ],

                                      "resp_ind": range(NUM_RESPONDENTS),
                                      "task_ind": range(NUM_TRAIN_TASKS)}

choice_and_demo_coords_with_interactions = {"visual_attributes": ["dialcolor",
                                                                  "dialshape",
                                                                  "strapcolor",
                                                                  "dialsize",
                                                                  "knobsize",
                                                                  "rimcolor",
                                                                  'dialcolor_dialshape', 'dialcolor_strapcolor',
                                                                   'dialcolor_dialsize', 'dialcolor_knobsize', 'dialcolor_rimcolor',
                                                                   'dialshape_strapcolor', 'dialshape_dialsize', 'dialshape_knobsize',
                                                                   'dialshape_rimcolor', 'strapcolor_dialsize', 'strapcolor_knobsize',
                                                                   'strapcolor_rimcolor', 'dialsize_knobsize', 'dialsize_rimcolor',
                                                                   'knobsize_rimcolor'],

                          "covariates": ["DemoGender_male",
                                         "DemoGender_female",
                                         "DemoAge_real",
                                         "DemoIncome_real",
                                         "DemoEducation_real",
                                         "DemoAestheticImportance1_1_real",
                                         # "DemoTimeSpentGrooming_real" # left out
                                         ],

                          "resp_ind": range(NUM_RESPONDENTS),
                          "task_ind": range(NUM_TRAIN_TASKS)}

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# -----------------------------------------------------------------------------------------------
#          Hierarchical Bayes Conjoint Model
#          Level 1: Choice Likelihood (Bernoulli)
#          Level 2: Population Prior - Mean (Respondent Covariats x Visual Attr)
#          Level 2: Population Prior - Full Covariance w/ Cholesky Decomp Sampling
#          Level 3: Hyperpriors - Including mean offset term
#          Defaults: GPU-Accel via Numpyro and Jax (all devices, change w/ CUDA_VISIBLE_DEVICES)
#          NOTE: July 26, 2023 CHANGE: Added support for pairwise interactions
# -----------------------------------------------------------------------------------------------

def run_HB_conjoint_model(X_train,
                          Y_train,
                          Z_df,
                          choice_and_demo_coords=None,
                          use_pairwise_interactions=True,
                          pairwise_interaction_std=None,
                          mu_theta_hyper_std=0.25,
                          theta_std=0.25,
                          beta_covar_dist=0.5,
                          beta_covar_eta=3.0,
                          num_draws=2000,
                          num_tune=2000,
                          num_chains=1,  # num of GPUs is most efficient for RAM -> VRAM
                          target_accept=0.65,
                          random_seed=0,
                          ):
    print("Using PyMC version: {}. Please ensure ver >= 5 for GPU support with Jax/Numpyro".format(pm.__version__))
    print("Using {} for MCMC sampling".format(jax.default_backend()))
    print("Sampling Devices: {}".format(jax.devices()))
    print("Note: No progress bar if > 1 GPU\n")

    if use_pairwise_interactions:
        assert pairwise_interaction_std is not None
        num_attributes = len(choice_and_demo_coords["visual_attributes"])
        X_train, X_train_interactions = X_train[:, :, :, :num_attributes], X_train[:, :, :, num_attributes:]
        print("****************************************\n              Using Pairwise Interactions              \n****************************************")
    else:
        print("****************************************\n              NOT Using Pairwise Interactions              \n****************************************")

    assert choice_and_demo_coords is not None
    num_attributes = len(choice_and_demo_coords["visual_attributes"])

    # Define HB Model - Wrap in self-context
    with pm.Model(coords=choice_and_demo_coords) as HB_ind_seg_model:

        #     X_attributes_left = pm.MutableData("X_attributes_left", X[:,:,0,:], dims=("resp_ind", "task_ind", "visual_attributes"))
        #     X_attributes_right = pm.MutableData("X_attributes_right", X[:,:,1,:], dims=("resp_ind", "task_ind", "visual_attributes"))
        X_attributes_left = pm.ConstantData("X_attributes_left",
                                            X_train[:, :, 0, :],
                                            dims=("resp_ind", "task_ind", "visual_attributes"))

        X_attributes_right = pm.ConstantData("X_attributes_right",
                                             X_train[:, :, 1, :],
                                             dims=("resp_ind", "task_ind", "visual_attributes"))

        if use_pairwise_interactions:
            X_interactions_left = pm.ConstantData("X_interactions_left",
                                                  X_train_interactions[:, :, 0, :],
                                                  dims=("resp_ind", "task_ind", "visual_attribute_interactions"))

            X_interactions_right = pm.ConstantData("X_interactions_right",
                                                   X_train_interactions[:, :, 1, :],
                                                   dims=("resp_ind", "task_ind", "visual_attribute_interactions"))

        covariates = pm.ConstantData("covariate_vars",
                                     Z_df.to_numpy(),
                                     dims=('resp_ind', 'covariates'))

        # Level 3: Hyperprior
        mu_theta_hyper = pm.Normal('beta_mu_hyper',
                                   mu=0,
                                   sigma=mu_theta_hyper_std,
                                   dims=('covariates', 'visual_attributes'))

        #  Level 2: Population Prior
        thetas = pm.Normal('thetas',
                           mu=mu_theta_hyper,
                           sigma=theta_std,
                           dims=('covariates', 'visual_attributes'))

        mu_beta = pm.Deterministic("mu_betas",
                                   pm.math.dot(covariates, thetas),
                                   dims=('resp_ind', 'visual_attributes'))

        # Covariance Matrix - LKJ Cholesky instead of Inverse-Wishart
        chol_beta, corr_beta, stds_beta = pm.LKJCholeskyCov("chol_beta",
                                                            n=num_attributes,
                                                            eta=beta_covar_eta,
                                                            sd_dist=pm.Exponential.dist(beta_covar_dist),
                                                            compute_corr=True)
        cov_beta = pm.Deterministic("cov_beta",
                                    chol_beta.dot(chol_beta.T))

        betas = pm.MvNormal("betas", mu_beta,
                            chol=chol_beta,
                            dims=('resp_ind', 'visual_attributes'))

        beta_mean = pm.Deterministic("beta_mean",
                                     betas.mean(axis=0))

        if use_pairwise_interactions:
            beta_interactions = pm.Normal('beta_interactions',
                                          mu=0,
                                          sigma=pairwise_interaction_std,
                                          dims=('visual_attribute_interactions'))

        # Level 1: Choice Likelihood (Bernoulli)
        U_left = (betas[:, None, :] * X_attributes_left).sum(axis=2)
        U_right = (betas[:, None, :] * X_attributes_right).sum(axis=2)

        if use_pairwise_interactions:
            U_left_interactions = (beta_interactions * X_interactions_left).sum(axis=2)
            U_right_interactions = (beta_interactions * X_interactions_right).sum(axis=2)
            U_left = U_left + U_left_interactions
            U_right = U_right + U_right_interactions

        p = pm.Deterministic("p", (pm.math.sigmoid(U_right) / (pm.math.sigmoid(U_left) + pm.math.sigmoid(U_right))), dims=("resp_ind", "task_ind"))

        # likelihood
        pm.Bernoulli("y",
                     p=p,
                     observed=Y_train,
                     #                  observed=Y,
                     #                 observed=valid_choices_and_demo_df["Response"],
                     dims=("resp_ind", "task_ind"))

        idata_seg = sampling_jax.sample_numpyro_nuts(draws=num_draws,
                                                     tune=num_tune,
                                                     chains=num_chains,
                                                     target_accept=target_accept,
                                                     random_seed=random_seed)
    return HB_ind_seg_model, idata_seg