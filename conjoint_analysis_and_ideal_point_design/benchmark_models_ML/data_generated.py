"""
#############################################################################################
    Project: Generative Interpretable Visual Design: Using Disentanglement for Visual Conjoint Analysis
    Author: Ankit Sisodia, Alex Burnap, Vineet Kumar
    Email: asisodia@purdue.edu, alex.burnap@yale.edu, vineet.kumar@yale.edu
    Date: July 2024
    License: MIT
#############################################################################################
"""

# Numerical and Plotting Libs
import numpy as np
import pandas as pd
# from PIL import Image
import matplotlib.pyplot as plt

# Neural Network Libs
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms #, datasets, models

# -----------------------------------------------------------------------------------------------
#                                 Globals
# -----------------------------------------------------------------------------------------------

PREDICTION_TASK = "generated_watches"

if PREDICTION_TASK == "real_watches":
    NUM_EPOCHS = 50
    IMG_W_H = 128
    NUM_DATA = 4261
    NUM_TRAIN = 3500
    BATCH_SIZE = 64
    NUM_RESPONDENTS = 286
    NUM_TASKS = 15
elif PREDICTION_TASK == "generated_watches":
    NUM_EPOCHS = 50
    IMG_W_H = 128
    NUM_RESPONDENTS = 256  # Note, this keeps in 3 respondents who said "Prefer not to say" for Gender - only w/o covariates
    NUM_RESPONDENTS = 253  # Note, this leaves out 3 respondents who said "Prefer not to say" for Gender
    NUM_ALTERNATIVES = 2
    NUM_ATTRIBUTES = 6
    NUM_TASKS = 15
    NUM_TRAIN_TASKS = 13
    NUM_TEST_TASKS = 2
    NUM_COVARIATES = 6
    # NUM_DATA = 3840 # Note, this keeps in 3 respondents who said "Prefer not to say" for Gender - only w/o covariates
    NUM_DATA = 3795 # Note, this leaves out 3 respondents who said "Prefer not to say" for Gender
    NUM_TRAIN = int(NUM_TRAIN_TASKS * NUM_RESPONDENTS)  # 13 out of 15 tasks per individual respondent
    BATCH_SIZE = 64
else:
    raise ValueError("Need to specify prediction task.")

COVARIATE_NAMES = ["DemoGender_male",
                    "DemoGender_female",
                    "DemoAge_real",
                    "DemoIncome_real",
                    "DemoEducation_real",
                    "DemoAestheticImportance1_1_real"]

# -----------------------------------------------------------------------------------------------
#                                 Directory Paths
# -----------------------------------------------------------------------------------------------
if PREDICTION_TASK == "real_watches":
    RAW_CONJOINT_PATH = "../../raw_data/conjoint_choice_data_real/"
    CBC_FILENAME = "CBCwatchexercise_choices_real_April042023.csv"
    CONJOINT_DATA_FILENAME = "Conjoint_Full_Data_Real_April042023.csv"
    PROCESSED_IMAGE_PATH = "../../processed_data/revised_conjoint_128x128/"
    EMBEDDED_REAL_WATCH_PATH = "../../raw_data/real_watch_embeddings_from_disent_model/real_watch_disent_and_vae_embeddings_and_mapping_04062023.csv"
elif PREDICTION_TASK == "generated_watches":
    RAW_CONJOINT_PATH = "../../raw_data/conjoint_choice_data_generated/"
    CBC_FILENAME = "CBCwatchexercise_simple_responses_generated_March820203.csv"
    CONJOINT_DATA_FILENAME = "FullConjointData_generated_mapped_variables_March172023.csv"
    PROCESSED_IMAGE_PATH = "../../processed_data/conjoint_gen_images/"
else:
    raise ValueError("Need to specify prediction task.")


# -----------------------------------------------------------------------------------------------
#                                 Get Data
# -----------------------------------------------------------------------------------------------

def get_data(use_thresholds=True):
    raw_conjoint_df = pd.read_csv(RAW_CONJOINT_PATH + CONJOINT_DATA_FILENAME,
                                  delimiter=",",
                                  header=0,
                                  index_col=0)
    raw_choices_df = pd.read_csv(RAW_CONJOINT_PATH + CBC_FILENAME,
                                 delimiter=",",
                                 header=0,
                                 index_col=0)
    if PREDICTION_TASK == "generated_watches":
        # raw_choices_df = raw_choices_df.copy()
        if use_thresholds:
            rescaling_dict = {1 : -1, 2 : 0, 3 : 1}
            raw_choices_df[["dialcolor", "dialshape", "strapcolor", "dialsize", "knobsize", "rimcolor"]] \
                = raw_choices_df[["dialcolor", "dialshape", "strapcolor", "dialsize", "knobsize", "rimcolor"]].replace(rescaling_dict)


            watch_thresholds = pd.read_csv("../../data_mappings/conj_gen_file_mapping_Thresholds.csv")
            dialcolor_dict = {key: value for key, value in zip([-1, 0, 1], watch_thresholds["DialColor"].unique())}
            dialshape_dict = {key: value for key, value in zip([-1, 0, 1], watch_thresholds["DialShape"].unique())}
            strapcolor_dict = {key: value for key, value in zip([-1, 0, 1], watch_thresholds["StrapColor"].unique())}
            dialsize_dict = {key: value for key, value in zip([-1, 0, 1], watch_thresholds["DialSize"].unique())}
            knobsize_dict = {key: value for key, value in zip([-1, 0, 1], watch_thresholds["KnobSize"].unique())}
            rimcolor_dict = {key: value for key, value in zip([-1, 0, 1], watch_thresholds["RimColor"].unique())}

            raw_choices_df["dialcolor"] = raw_choices_df["dialcolor"].map(dialcolor_dict)
            raw_choices_df["dialshape"] = raw_choices_df["dialshape"].map(dialshape_dict)
            raw_choices_df["strapcolor"] = raw_choices_df["strapcolor"].map(strapcolor_dict)
            raw_choices_df["dialsize"] = raw_choices_df["dialsize"].map(dialsize_dict)
            raw_choices_df["knobsize"] = raw_choices_df["knobsize"].map(knobsize_dict)
            raw_choices_df["rimcolor"] = raw_choices_df["rimcolor"].map(rimcolor_dict)

        else:
            rescaling_dict = {1: "L", 2: "M", 3: "H"}
            raw_choices_df[["dialcolor", "dialshape", "strapcolor", "dialsize", "knobsize", "rimcolor"]] \
                = raw_choices_df[["dialcolor", "dialshape", "strapcolor", "dialsize", "knobsize", "rimcolor"]].replace(rescaling_dict)

        return raw_conjoint_df, raw_choices_df
    if PREDICTION_TASK == "real_watches":
        embedded_real_df_raw = pd.read_csv(EMBEDDED_REAL_WATCH_PATH)
        return raw_conjoint_df, raw_choices_df, embedded_real_df_raw


def create_X_Y_full_matrices(raw_choices_df):
    X_full = np.zeros([NUM_RESPONDENTS * NUM_TASKS, 2])
    Y_full = np.zeros([NUM_RESPONDENTS * NUM_TASKS])
    for ind, resp_ind in enumerate(raw_choices_df.index.unique().values):
        X_full[ind * NUM_TASKS:(ind * NUM_TASKS + NUM_TASKS), 0] = raw_choices_df.loc[resp_ind]["realwatchid"][0::2].values
        X_full[ind * NUM_TASKS:(ind * NUM_TASKS + NUM_TASKS), 1] = raw_choices_df.loc[resp_ind]["realwatchid"][1::2].values
        Y_full[ind * NUM_TASKS:(ind * NUM_TASKS + NUM_TASKS)] = raw_choices_df.loc[resp_ind]["Response"][1::2]

    # Drop all choices that contained "realimage_154 since corrupted..."
    Y_full = np.delete(Y_full, np.where(X_full == 155)[0], axis=0)
    X_full = np.delete(X_full, np.where(X_full == 155)[0], axis=0)
    return X_full, Y_full

def create_X_Y_generated_full_matrices(raw_choices_df):
    X = np.zeros([NUM_RESPONDENTS, NUM_TASKS, NUM_ALTERNATIVES, NUM_ATTRIBUTES], dtype="str")
    Y = np.zeros([NUM_RESPONDENTS, NUM_TASKS])

    for x_ind, resp_ind in enumerate(raw_choices_df.index.unique().values):
        for task_ind in range(NUM_TASKS):
            X[x_ind, task_ind, 0, :] = raw_choices_df.loc[resp_ind][raw_choices_df.loc[resp_ind]["Task"]==task_ind+1][["dialcolor", "dialshape", "strapcolor", "dialsize", "knobsize", "rimcolor"]].iloc[0]
            X[x_ind, task_ind, 1, :] = raw_choices_df.loc[resp_ind][raw_choices_df.loc[resp_ind]["Task"]==task_ind+1][["dialcolor", "dialshape", "strapcolor", "dialsize", "knobsize", "rimcolor"]].iloc[1]
            Y[x_ind, task_ind] = raw_choices_df.loc[resp_ind][raw_choices_df.loc[resp_ind]["Task"]==task_ind+1][["Response"]].iloc[1]

    X = X.reshape(NUM_RESPONDENTS * NUM_TASKS, NUM_ALTERNATIVES, NUM_ATTRIBUTES)
    Y = Y.reshape(NUM_RESPONDENTS * NUM_TASKS)

    return X, Y

def create_X_Y_Z_generated_full_matrices(raw_choices_df, raw_conjoint_df):
    # X = np.zeros([NUM_RESPONDENTS, NUM_TASKS, NUM_ALTERNATIVES, NUM_ATTRIBUTES], dtype="str")
    X = np.zeros([NUM_RESPONDENTS, NUM_TASKS, NUM_ALTERNATIVES, NUM_ATTRIBUTES], dtype=np.float32)
    Y = np.zeros([NUM_RESPONDENTS, NUM_TASKS])
    # Z = np.zeros([NUM_RESPONDENTS, NUM_COVARIATES])
    Z = np.zeros([NUM_RESPONDENTS, NUM_TASKS, NUM_COVARIATES])

    valid_users_array = raw_choices_df.index.unique().values
    raw_conjoint_df = raw_conjoint_df[raw_conjoint_df.index.isin(valid_users_array)]
    raw_conjoint_df = raw_conjoint_df[['DemoGender', 'DemoAge_real', 'DemoEducation_real',
                                               'DemoIncome_real', 'DemoSocialMedia_real',
                                               'DemoAestheticImportance1_1_real', 'DemoAestheticImportance2_1_real',
                                               'DemoTimeSpentGrooming_real', 'DemoOwnWatch_real', 'DemoWatchWTP_real']]
    valid_choices_and_demo_df = raw_choices_df.join(other=raw_conjoint_df,
                                                         on="sys_RespNum",
                                                         how="left")
    # Drop 6 people who put Non-Binary or Prefer not to say for Gender
    valid_choices_and_demo_df = valid_choices_and_demo_df[(valid_choices_and_demo_df["DemoGender"] == 1) | (valid_choices_and_demo_df["DemoGender"] == 2)]
    # Convert DemoGender to one hot
    valid_choices_and_demo_df["DemoGender_male"] = valid_choices_and_demo_df["DemoGender"]
    valid_choices_and_demo_df["DemoGender_female"] = valid_choices_and_demo_df["DemoGender"]

    valid_choices_and_demo_df["DemoGender_male"] = valid_choices_and_demo_df["DemoGender_male"].apply(lambda x: 0 if x == 2 else 1)
    valid_choices_and_demo_df["DemoGender_female"] = valid_choices_and_demo_df["DemoGender_female"].apply(lambda x: 1 if x == 2 else 0)
    ## Renormalize all real-valued to min-max scaling in [-1,1]
    valid_choices_and_demo_df[COVARIATE_NAMES] = valid_choices_and_demo_df[COVARIATE_NAMES] / valid_choices_and_demo_df[COVARIATE_NAMES].max()

    for x_ind, resp_ind in enumerate(valid_choices_and_demo_df.index.unique().values):

        for task_ind in range(NUM_TASKS):
            X[x_ind, task_ind, 0, :] = valid_choices_and_demo_df.loc[resp_ind][valid_choices_and_demo_df.loc[resp_ind]["Task"]==task_ind+1][["dialcolor", "dialshape", "strapcolor", "dialsize", "knobsize", "rimcolor"]].iloc[0]
            X[x_ind, task_ind, 1, :] = valid_choices_and_demo_df.loc[resp_ind][valid_choices_and_demo_df.loc[resp_ind]["Task"]==task_ind+1][["dialcolor", "dialshape", "strapcolor", "dialsize", "knobsize", "rimcolor"]].iloc[1]
            Y[x_ind, task_ind] = valid_choices_and_demo_df.loc[resp_ind][valid_choices_and_demo_df.loc[resp_ind]["Task"]==task_ind+1][["Response"]].iloc[1]
            Z[x_ind, task_ind, :] = valid_choices_and_demo_df.loc[resp_ind][COVARIATE_NAMES].iloc[0]

    X = X.reshape(NUM_RESPONDENTS * NUM_TASKS, NUM_ALTERNATIVES, NUM_ATTRIBUTES)
    Y = Y.reshape(NUM_RESPONDENTS * NUM_TASKS)
    Z = Z.reshape(NUM_RESPONDENTS * NUM_TASKS, NUM_COVARIATES)

    return X, Y, Z

def create_embedding_X_matrix(X_full, embedded_real_df_raw=None):
    mapping_to_conjoint = embedded_real_df_raw["conjoint_file_name"].apply(lambda x: x.split("_")[1].split(".")[0]).astype(int)
    embedded_real_df_raw_sorted = embedded_real_df_raw.loc[np.argsort(mapping_to_conjoint.values)]
    embedded_real_df_raw_sorted.index = range(len(embedded_real_df_raw_sorted))
    embedded_real_matrix = embedded_real_df_raw_sorted[["Dial_Color", "Dial_Shape", "Strap_Color", "Dial_Size", "Knob_Size", "Rim_Color"]]

    x_full_embeddings = np.zeros([X_full.shape[0], 2, 6], dtype=np.float32)

    for ind in range(X_full.shape[0]):
        # Note: We convert from Sawtooth 1 indexing to 0 indexed Python
        img_ind_left, img_ind_right = int(X_full[ind, 0] - 1), int(X_full[ind, 1] - 1)
        x_full_embeddings[ind, 0, :] = embedded_real_matrix.loc[img_ind_left].values
        x_full_embeddings[ind, 1, :] = embedded_real_matrix.loc[img_ind_right].values

    return x_full_embeddings

def create_entangled_embedding_X_matrix(X_full, embedded_real_df_raw=None):
    mapping_to_conjoint = embedded_real_df_raw["conjoint_file_name"].apply(lambda x: x.split("_")[1].split(".")[0]).astype(int)
    embedded_real_df_raw_sorted = embedded_real_df_raw.loc[np.argsort(mapping_to_conjoint.values)]
    embedded_real_df_raw_sorted.index = range(len(embedded_real_df_raw_sorted))
    embedded_real_matrix = embedded_real_df_raw_sorted[["vae_1","vae_2", "vae_3", "vae_4", "vae_5", "vae_6"]]

    x_full_embeddings = np.zeros([X_full.shape[0], 2, 6], dtype=np.float32)

    for ind in range(X_full.shape[0]):
        # Note: We convert from Sawtooth 1 indexing to 0 indexed Python
        img_ind_left, img_ind_right = int(X_full[ind, 0] - 1), int(X_full[ind, 1] - 1)
        x_full_embeddings[ind, 0, :] = embedded_real_matrix.loc[img_ind_left].values
        x_full_embeddings[ind, 1, :] = embedded_real_matrix.loc[img_ind_right].values

    return x_full_embeddings


def create_image_X_matrix(X_full):
    X_full_images = np.ones([X_full.shape[0], 2, 128, 128, 3], dtype=np.uint8)

    for ind in range(X_full.shape[0]):
        # Note: We convert from Sawtooth 1 indexing to 0 indexed Python
        img_ind_left, img_ind_right = int(X_full[ind, 0] - 1), int(X_full[ind, 1] - 1)
        X_full_images[ind, 0, :, :, :] = plt.imread(PROCESSED_IMAGE_PATH + 'realwatch_{}.jpg'.format(img_ind_left))
        X_full_images[ind, 1, :, :, :] = plt.imread(PROCESSED_IMAGE_PATH + 'realwatch_{}.jpg'.format(img_ind_right))

    return X_full_images


def create_generated_image_X_matrix(X_full):
    X_full_images = np.ones([X_full.shape[0], 2, 128, 128, 3], dtype=np.uint8)

    for ind in range(X_full.shape[0]):
        img_ind_left, img_ind_right = X_full[ind, 0, :], X_full[ind, 1, :]

        X_full_images[ind, 0, :, :, :] = plt.imread(PROCESSED_IMAGE_PATH + 'dialcolor_{}_dialshape_{}_strapcolor_{}_dialsize_{}_knobsize_{}_rimcolor_{}.jpg'.format(*img_ind_left))
        X_full_images[ind, 1, :, :, :] = plt.imread(PROCESSED_IMAGE_PATH + 'dialcolor_{}_dialshape_{}_strapcolor_{}_dialsize_{}_knobsize_{}_rimcolor_{}.jpg'.format(*img_ind_right))

    return X_full_images

#-----------------------------------------------------------------------------------------------
#                                 Dataset Classes
#-----------------------------------------------------------------------------------------------

class WatchDatasetEmbedded(Dataset):
    def __init__(self,
                 X_full_embeddings=None,
                 Y_full=None,
                 Z_full=None):
        assert X_full_embeddings is not None
        assert Y_full is not None

        if Z_full is not None:
            self.Z_full = Z_full.astype(np.float32)
            self.use_covariates = True
        else:
            self.use_covariates = False

        self.X_full_embeddings = X_full_embeddings
        self.Y_full = Y_full

    def __len__(self):
        return len(self.X_full_embeddings)

    def __getitem__(self, idx):
        embedding_left = self.X_full_embeddings[idx, 0, :]
        embedding_right = self.X_full_embeddings[idx, 1, :]
        y = self.Y_full[idx]

        if not self.use_covariates:
            return embedding_left, embedding_right, y
        else:
            z = self.Z_full[idx]
            return embedding_left, embedding_right, y, z

# class WatchDatasetEmbedded(Dataset):
#     def __init__(self,
#                  X_full_embeddings=None,
#                  Y_full=None):
#         assert X_full_embeddings is not None
#         assert Y_full is not None
#
#         self.X_full_embeddings = X_full_embeddings
#         self.Y_full = Y_full
#
#     def __len__(self):
#         return len(self.X_full_embeddings)
#
#     def __getitem__(self, idx):
#         embedding_left = self.X_full_embeddings[idx, 0, :]
#         embedding_right = self.X_full_embeddings[idx, 1, :]
#         y = self.Y_full[idx]
#
#         return embedding_left, embedding_right, y


class WatchDataset(Dataset):
    def __init__(self,
                 X_full_images=None,
                 Y_full=None,
                 Z_full=None,
                 subset=False,
                 custom_transforms=None):

        assert X_full_images is not None
        assert Y_full is not None

        self.X_full_images = X_full_images.astype(np.float32) / 255.0
        self.Y_full = Y_full.astype(np.float32)
        if Z_full is not None:
            self.Z_full = Z_full.astype(np.float32)
            self.use_covariates = True
        else:
            self.Z_full = None
            self.use_covariates = False

        # Data Transforms
        if custom_transforms:
            self.transforms = custom_transforms
        else:
            #             self.transforms = None
            normalize_inputs = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]
                                                    )
            #             self.transform_images = transforms.Compose([transforms.ToTensor(),
            #                                                         normalize_inputs,
            #                                                         transforms.Resize([224,224], antialias=True),
            # #                                                         transforms.CenterCrop(224)
            #                                                        ])

            self.transform_images = transforms.Compose([transforms.ToTensor(),
                                                        transforms.Resize([224, 224], antialias=True),
                                                        normalize_inputs,
                                                        #                                                         transforms.CenterCrop(224)
                                                        ])

    def __len__(self):
        return len(self.X_full_images)

    def __getitem__(self, idx):

        image_left = self.X_full_images[idx, 0, :, :, :]
        image_right = self.X_full_images[idx, 1, :, :, :]
        y = self.Y_full[idx]

        image_left = self.transform_images(image_left)  # .permute(2, 0, 1)
        image_right = self.transform_images(image_right)  # .permute(2, 0, 1)
        if not self.use_covariates:
            return image_left, image_right, y
        else:
            z = self.Z_full[idx]
            return image_left, image_right, y, z
