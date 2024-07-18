"""
#############################################################################################
                Project: JMR23 - Disentangling Watches
                Author: Alex Burnap
                Email: alex.burnap@yale.edu
                Date: Feb. 2, 2024
                License: MIT
#############################################################################################
"""
# Standard Libs
import os

# Numerical and Plotting Libs - None

# Bayesian Libs
import pymc as pm
import jax
import pymc.sampling_jax as sampling_jax
import numpy as np

# -----------------------------------------------------------------------------------------------
#                                 Globals
# -----------------------------------------------------------------------------------------------

NUM_RESPONDENTS = 253
NUM_ALTERNATIVES = 2
NUM_ATTRIBUTES = 6
NUM_TASKS = 15
NUM_TRAIN_TASKS = 13
NUM_TEST_TASKS = 2
# NUM_COVARIATES = 6
NUM_COVARIATES = 7

choice_and_demo_coords = {"visual_attributes": ["dialcolor",
                                                "dialshape",
                                                "strapcolor",
                                                "dialsize",
                                                "knobsize",
                                                "rimcolor"],

                          "covariates": ["const_value",
                              "DemoGender_male",
                              "DemoGender_female",
                              "DemoAge_real",
                              "DemoIncome_real",
                              "DemoEducation_real",
                              "DemoAestheticImportance1_1_real",
                              ],


                          "resp_ind": range(NUM_RESPONDENTS),
                          "task_ind": range(NUM_TRAIN_TASKS)}

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# -----------------------------------------------------------------------------------------------
#          Hierarchical Bayes Conjoint Model
#          Level 1: Choice Likelihood (Bernoulli)
#          Level 2: Population Prior - Mean (Respondent Covariates x Visual Attr) + Intercept (Visual Attr x 1)
#          Level 2: Population Prior - Full Covariance w/ Cholesky Decomp Sampling
#          -------: (Deprecated) Hyperpriors - Theta Intercept as Mean Offset Term --> Converted to more standard 2-level model w/ 1's appended to Z
#          Defaults: GPU-Accel via Numpyro and Jax (1 device, change to Multi-GPU w/ CUDA_VISIBLE_DEVICES)
#                    Single GPU replicates paper exactly.  Multi-GPU is nondeterministic.
# -----------------------------------------------------------------------------------------------

def run_HB_conjoint_model(X_train,
                          Y_train,
                          Z_df,
                          theta_std=0.25,
                          beta_covar_dist=0.5,
                          beta_covar_eta=4.0,
                          num_draws=2000,
                          num_tune=2000,
                          num_chains=1, # num of GPUs is most efficient for RAM -> VRAM
                          target_accept=0.65,
                          random_seed=0,
                          ):

    print("Using PyMC version: {}. Please ensure ver >= 5 for GPU support with Jax/Numpyro".format(pm.__version__))
    print("Using {} for MCMC sampling".format(jax.default_backend()))
    print("Sampling Devices: {}".format(jax.devices()))
    print("Note: No progress bar if > 1 GPU")

    # Define HB Model - Wrap in self-context
    with pm.Model(coords=choice_and_demo_coords) as HB_ind_seg_model:

        X_attributes_left = pm.ConstantData("X_attributes_left",
                                            X_train[:, :, 0, :],
                                            dims=("resp_ind", "task_ind", "visual_attributes"))

        X_attributes_right = pm.ConstantData("X_attributes_right",
                                             X_train[:, :, 1, :],
                                             dims=("resp_ind", "task_ind", "visual_attributes"))

        covariates = pm.ConstantData("covariate_vars",
                np.hstack([np.ones((Z_df.shape[0], 1)), Z_df.to_numpy()]).T,
                dims=('covariates', 'resp_ind'))

        # Main change from prior open source code release - Changed to more standard formulation of \Theta_0 intercept term
        # Specifically: Removing explicit hyperprior intercept term in lieu of adding a vector of 1's to the covariate matrix Z
        # # Level 3: Hyperprior
        # mu_theta_hyper = pm.Normal('theta_mu_hyper',
        #                            mu=0,
        #                            sigma=mu_theta_hyper_std,
        #                            dims=('visual_attributes', 'covariates'))

        #  Level 2: Population Prior
        thetas = pm.Normal('thetas',
                           # mu=mu_theta_hyper,
                           mu=0,
                           sigma=theta_std,
                           dims=('visual_attributes', 'covariates'))

        mu_beta = pm.Deterministic("mu_betas",
                                   pm.math.dot(thetas, covariates).T,
                                   dims=('resp_ind', 'visual_attributes'))

        # Covariance Matrix - LKJ Cholesky instead of Inverse-Wishart
        chol_beta, corr_beta, stds_beta = pm.LKJCholeskyCov("chol_beta",
                                                            n=6,
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

        # Level 1: Choice Likelihood (Bernoulli)
        U_left = (betas[:, None, :] * X_attributes_left).sum(axis=2)
        U_right = (betas[:, None, :] * X_attributes_right).sum(axis=2)

        p = pm.Deterministic("p", (pm.math.sigmoid(U_right) / (pm.math.sigmoid(U_left) + pm.math.sigmoid(U_right))), dims=("resp_ind", "task_ind"))

        # likelihood
        pm.Bernoulli("y",
                     p=p,
                     observed=Y_train,
                     dims=("resp_ind", "task_ind"))

        idata_seg = sampling_jax.sample_numpyro_nuts(draws=num_draws,
                                                     tune=num_tune,
                                                     chains=num_chains,
                                                     target_accept=target_accept,
                                                     random_seed=random_seed)
    return HB_ind_seg_model, idata_seg
