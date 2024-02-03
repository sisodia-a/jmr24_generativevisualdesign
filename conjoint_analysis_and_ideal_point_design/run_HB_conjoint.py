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

# -----------------------------------------------------------------------------------------------
#                                 Globals
# -----------------------------------------------------------------------------------------------

NUM_RESPONDENTS = 253
NUM_ALTERNATIVES = 2
NUM_ATTRIBUTES = 6
NUM_TASKS = 15
NUM_TRAIN_TASKS = 13
NUM_TEST_TASKS = 2
NUM_COVARIATES = 6

choice_and_demo_coords = {"visual_attributes": ["dialcolor",
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
                                         "DemoAestheticImportance1_1_real"
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
# -----------------------------------------------------------------------------------------------

def run_HB_conjoint_model(X_train,
                          Y_train,
                          Z_df,
                          mu_theta_hyper_std=1.5,
                          theta_std=1.5,
                          beta_covar_dist=1.5,
                          beta_covar_eta=6.0,
                          num_draws=5000,
                          num_tune=5000,
                          num_chains=4, # num of GPUs is most efficient for RAM -> VRAM
                          target_accept=0.8,
                          random_seed=0,
                          ):

    print("Using PyMC version: {}. Please ensure ver >= 5 for GPU support with Jax/Numpyro".format(pm.__version__))
    print("Using {} for MCMC sampling".format(jax.default_backend()))
    print("Sampling Devices: {}".format(jax.devices()))
    print("Note: No progress bar if > 1 GPU")

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

        covariates = pm.ConstantData("covariate_vars",
                                     Z_df.to_numpy().T,
                                     # dims=('resp_ind', 'covariates'))
                                     dims = ('covariates', 'resp_ind'))

        # Level 3: Hyperprior
        mu_theta_hyper = pm.Normal('theta_mu_hyper',
                                   mu=0,
                                   sigma=mu_theta_hyper_std,
                                   # dims=('covariates', 'visual_attributes'))
                                   dims=('visual_attributes', 'covariates'))

        #  Level 2: Population Prior
        thetas = pm.Normal('thetas',
                           mu=mu_theta_hyper,
                           sigma=theta_std,
                           # dims=('covariates', 'visual_attributes')
                           dims=('visual_attributes', 'covariates'))

        mu_beta = pm.Deterministic("mu_betas",
                                   # pm.math.dot(covariates, thetas),
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
                     #                  observed=Y,
                     #                 observed=valid_choices_and_demo_df["Response"],
                     dims=("resp_ind", "task_ind"))

        idata_seg = sampling_jax.sample_numpyro_nuts(draws=num_draws,
                                                     tune=num_tune,
                                                     chains=num_chains,
                                                     target_accept=target_accept,
                                                     random_seed=random_seed)
    return HB_ind_seg_model, idata_seg
