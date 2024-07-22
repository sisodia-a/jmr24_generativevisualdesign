"""
#############################################################################################
    Project: Generative Interpretable Visual Design: Using Disentanglement for Visual Conjoint Analysis
    Author: Ankit Sisodia, Alex Burnap, Vineet Kumar
    Email: asisodia@purdue.edu, alex.burnap@yale.edu, vineet.kumar@yale.edu
    Date: July 2024
    License: MIT
#############################################################################################
"""
# Standard Libs - None

# Numerical and Plotting Libs
import numpy as np
import pandas as pd

# Bayesian Libs - None

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


COVARIATE_NAMES = ["DemoGender_male",
                    "DemoGender_female",
                    "DemoAge_real",
                    "DemoIncome_real",
                    "DemoEducation_real",
                    "DemoAestheticImportance1_1_real"]


VISUAL_ATTRIBUTE_NAMES =  ["dialcolor",
                         "dialshape",
                         "strapcolor",
                         "dialsize",
                         "knobsize",
                         "rimcolor"]

# -----------------------------------------------------------------------------------------------
#                                 Directory Paths
# -----------------------------------------------------------------------------------------------

RAW_CONJOINT_PATH = "./data_conjointanalysis/"
CONJOINT_RESPONSES_FILENAME = "CBCwatchexercise_simple_responses_generated.csv"
CONJOINT_DATA_FILENAME = "FullConjointData_generated_mapped_variables.csv"
DATA_MAPPING_01_TO_POSTERIOR_PATH = "conj_gen_file_mapping_Thresholds.csv"

# -----------------------------------------------------------------------------------------------
#                                 Get Data
# -----------------------------------------------------------------------------------------------

def get_data(TRAIN_TASK_INDICES, TEST_TASK_INDICES):
    raw_conjoint_df = pd.read_csv(RAW_CONJOINT_PATH + CONJOINT_DATA_FILENAME,
                                  delimiter=",",
                                  header=0,
                                  index_col=0)

    raw_choices_df = pd.read_csv(RAW_CONJOINT_PATH + CONJOINT_RESPONSES_FILENAME,
                                 delimiter=",",
                                 header=0,
                                 index_col=0)

    # raw_choices_df already has respondent filtering
    valid_users_array = raw_choices_df.index.unique().values
    valid_conjoint_df = raw_conjoint_df[raw_conjoint_df.index.isin(valid_users_array)]

    # Rescale to -1, 0, 1 -- later changed to posterior threshold mapping (to be consistent with image generation thresholds)
    choices_rescaled_df = raw_choices_df.copy()
    rescaling_dict = {1 : -1, 2 : 0, 3 : 1}
    choices_rescaled_df[["dialcolor", "dialshape", "strapcolor", "dialsize", "knobsize", "rimcolor"]] \
        = choices_rescaled_df[["dialcolor", "dialshape", "strapcolor", "dialsize", "knobsize", "rimcolor"]].replace(rescaling_dict)

    # Map Demographics/Seg Vars to Respondents
    valid_demographics_df = valid_conjoint_df[['DemoGender', 'DemoAge_real', 'DemoEducation_real',
                                               'DemoIncome_real', 'DemoSocialMedia_real',
                                               'DemoAestheticImportance1_1_real', 'DemoAestheticImportance2_1_real',
                                               'DemoTimeSpentGrooming_real', 'DemoOwnWatch_real', 'DemoWatchWTP_real']]

    # Join DataFrames
    valid_choices_and_demo_df = choices_rescaled_df.join(other=valid_demographics_df,
                                                         on="sys_RespNum",
                                                         how="left")

    # Drop 6 people who put Non-Binary or Prefer not to say for Gender
    valid_choices_and_demo_df = valid_choices_and_demo_df[(valid_choices_and_demo_df["DemoGender"] == 1) | (valid_choices_and_demo_df["DemoGender"] == 2)]

    # Convert DemoGender to one hot
    valid_choices_and_demo_df["DemoGender_male"] = valid_choices_and_demo_df["DemoGender"]
    valid_choices_and_demo_df["DemoGender_female"] = valid_choices_and_demo_df["DemoGender"]

    valid_choices_and_demo_df["DemoGender_male"] = valid_choices_and_demo_df["DemoGender_male"].apply(lambda x: 0 if x == 2 else 1)
    valid_choices_and_demo_df["DemoGender_female"] = valid_choices_and_demo_df["DemoGender_female"].apply(lambda x: 1 if x == 2 else 0)

    # Save sys_RespNum for mapping to Sawtooth Conjoint Survey - prep for Reindex to consecutive array
    valid_choices_and_demo_df['sys_RespNum'] = valid_choices_and_demo_df.index

    # Renormalize all real-valued covariates to min-max scaling in [-1,1]
    valid_choices_and_demo_df[COVARIATE_NAMES] = valid_choices_and_demo_df[COVARIATE_NAMES] / valid_choices_and_demo_df[COVARIATE_NAMES].max()

    # Map Posterior thresholds to -1, 0, 1 data... note also requires mapping inconsistent viz char names
    mapping_thresholds = pd.read_csv(RAW_CONJOINT_PATH + DATA_MAPPING_01_TO_POSTERIOR_PATH)
    dialcolor_dict = {key: value for key, value in zip([-1, 0, 1], mapping_thresholds["DialColor"].unique())}
    dialshape_dict = {key: value for key, value in zip([-1, 0, 1], mapping_thresholds["DialShape"].unique())}
    strapcolor_dict = {key: value for key, value in zip([-1, 0, 1], mapping_thresholds["StrapColor"].unique())}
    dialsize_dict = {key: value for key, value in zip([-1, 0, 1], mapping_thresholds["DialSize"].unique())}
    knobsize_dict = {key: value for key, value in zip([-1, 0, 1], mapping_thresholds["KnobSize"].unique())}
    rimcolor_dict = {key: value for key, value in zip([-1, 0, 1], mapping_thresholds["RimColor"].unique())}
    valid_choices_and_demo_df["dialcolor"] = valid_choices_and_demo_df["dialcolor"].map(dialcolor_dict)
    valid_choices_and_demo_df["dialshape"] = valid_choices_and_demo_df["dialshape"].map(dialshape_dict)
    valid_choices_and_demo_df["strapcolor"] = valid_choices_and_demo_df["strapcolor"].map(strapcolor_dict)
    valid_choices_and_demo_df["dialsize"] = valid_choices_and_demo_df["dialsize"].map(dialsize_dict)
    valid_choices_and_demo_df["knobsize"] = valid_choices_and_demo_df["knobsize"].map(knobsize_dict)
    valid_choices_and_demo_df["rimcolor"] = valid_choices_and_demo_df["rimcolor"].map(rimcolor_dict)

    # Populate X, Y, Z matrices
    X_full = np.zeros([NUM_RESPONDENTS, NUM_TASKS, NUM_ALTERNATIVES, NUM_ATTRIBUTES])
    Y_full = np.zeros([NUM_RESPONDENTS, NUM_TASKS])
    Z = np.zeros([NUM_RESPONDENTS, NUM_COVARIATES])

    for x_ind, resp_ind in enumerate(valid_choices_and_demo_df.index.unique().values):
        Z[x_ind, :] = valid_choices_and_demo_df.loc[resp_ind][COVARIATE_NAMES].iloc[0]
        for task_ind in range(NUM_TASKS):
            X_full[x_ind, task_ind, 0, :] = valid_choices_and_demo_df.loc[resp_ind][valid_choices_and_demo_df.loc[resp_ind]["Task"] == task_ind + 1][["dialcolor", "dialshape", "strapcolor", "dialsize", "knobsize", "rimcolor"]].iloc[0]
            X_full[x_ind, task_ind, 1, :] = valid_choices_and_demo_df.loc[resp_ind][valid_choices_and_demo_df.loc[resp_ind]["Task"] == task_ind + 1][["dialcolor", "dialshape", "strapcolor", "dialsize", "knobsize", "rimcolor"]].iloc[1]
            Y_full[x_ind, task_ind] = valid_choices_and_demo_df.loc[resp_ind][valid_choices_and_demo_df.loc[resp_ind]["Task"] == task_ind + 1][["Response"]].iloc[1]

    X_train = X_full[:, TRAIN_TASK_INDICES, :, :]
    X_test = X_full[:, TEST_TASK_INDICES, :, :]
    Y_train = Y_full[:, TRAIN_TASK_INDICES]
    Y_test = Y_full[:, TEST_TASK_INDICES]

    Y_train_df = pd.DataFrame(data=Y_train)
    Y_test_df = pd.DataFrame(data=Y_test)
    Z_df = pd.DataFrame(data=Z, columns=COVARIATE_NAMES, index=range(NUM_RESPONDENTS))

    return X_train, X_test, Y_train, Y_test, Z_df
