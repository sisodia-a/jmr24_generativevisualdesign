"""
#############################################################################################
                Project: JMR23 - Disentangling Watches
                Author: Alex Burnap
                Email: alex.burnap@yale.edu
                Date: April 6, 2023
                License: MIT
#############################################################################################
"""
import os
import math
import glob
import time

# Numerical and Plotting Libs
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Neural Network Libs
import torch
from torch import nn
import torch.nn.parallel
import torch.optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms #, datasets, models


class PretrainedBenchmarkModel(nn.Module):

    def __init__(self,
                 pretrained_model_arch="resnet50",
                 #                  pretrained_model_arch = "VGG16",
                 tower_fc_units=64,
                 use_cuda=True):
        super(PretrainedBenchmarkModel, self).__init__()

        self.tower_fc_units = tower_fc_units
        self.pretrained_model_arch = pretrained_model_arch

        # Download Pretrained Model
        if self.pretrained_model_arch == "resnet50":

            self.pretrained_model = torchvision.models.resnet50(pretrained=True)
            self.embedding_size = self.pretrained_model.fc.in_features

        elif self.pretrained_model_arch == "VGG16":
            self.pretrained_model = torchvision.models.vgg16(pretrained=True)
            #             self.pretrained_model = self.pretrained_model.features
            #             self.embedding_size = 8192
            #             self.embedding_size = 25088
            self.embedding_size = 4096

        else:
            raise ValueError("arch not found dangit")

        #             resnet_weights = ResNet50_Weights.DEFAULT
        #             self.preprocess_transform = resnet_weights.transforms()
        #             normalize_inputs = transforms.Normalize(
        #                                                mean=[0.485, 0.456, 0.406],
        #                                                std=[0.229, 0.224, 0.225]
        #                                             )

        # Freeze Pretrained Model Weights
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        # Remap FC layers to our tower
        #         self.pretrained_model.fc = nn.Sequential(
        #                                 nn.Linear(self.embedding_size, self.tower_fc_units),
        #                                 nn.ReLU(inplace=True),
        # #                                 nn.Linear(self.tower_fc_units, self.tower_fc_units),
        # #                                 nn.ReLU(inplace=True),
        #                                 nn.Linear(self.tower_fc_units, 1),
        #                                 nn.Sigmoid()
        #                                 )

        if self.pretrained_model_arch == "resnet50":
            self.pretrained_model.fc = nn.Sequential(

                #                         nn.Linear(self.embedding_size, 1),
                nn.Linear(self.embedding_size, self.tower_fc_units),
                nn.ReLU(inplace=True),
                nn.Linear(self.tower_fc_units, self.tower_fc_units),
                nn.ReLU(inplace=True),
                nn.Linear(self.tower_fc_units, 1),
                nn.Sigmoid()
            )
        elif self.pretrained_model_arch == "VGG16":
            self.pretrained_model.classifier[6] = nn.Sequential(

                nn.Linear(self.embedding_size, 1),
                nn.Linear(self.embedding_size, self.tower_fc_units),
                nn.ReLU(inplace=True),
                nn.Linear(self.embedding_size, self.tower_fc_units),
                nn.ReLU(inplace=True),
                nn.Linear((self.tower_fc_units), 1),
                nn.Sigmoid()
            )


    def forward(self, image_left, image_right):

        logit_left = self.pretrained_model(image_left)
        logit_right = self.pretrained_model(image_right)

        p = logit_right / (logit_left + logit_right)

        return p


class PretrainedWithCovariatesBenchmarkModel(nn.Module):

    def __init__(self,
                 pretrained_model_arch="resnet50",
                 #                  pretrained_model_arch = "VGG16",
                 tower_fc_units=64,
                 num_covariates=6,
                 use_cuda=True):
        super(PretrainedWithCovariatesBenchmarkModel, self).__init__()

        self.tower_fc_units = tower_fc_units
        self.pretrained_model_arch = pretrained_model_arch
        self.num_covariates = num_covariates

        # Download Pretrained Model
        if self.pretrained_model_arch == "resnet50":

            self.pretrained_model = torchvision.models.resnet50(pretrained=True)
            self.embedding_size = self.pretrained_model.fc.in_features# + self.num_covariates

        elif self.pretrained_model_arch == "VGG16":
            self.pretrained_model = torchvision.models.vgg16(pretrained=True)
            #             self.pretrained_model = self.pretrained_model.features
            #             self.embedding_size = 8192
            #             self.embedding_size = 25088
            self.embedding_size = 4096# + self.num_covariates

        else:
            raise ValueError("arch not found dangit")

        #             resnet_weights = ResNet50_Weights.DEFAULT
        #             self.preprocess_transform = resnet_weights.transforms()
        #             normalize_inputs = transforms.Normalize(
        #                                                mean=[0.485, 0.456, 0.406],
        #                                                std=[0.229, 0.224, 0.225]
        #                                             )

        # Freeze Pretrained Model Weights
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        # Remap FC layers to our tower
        #         self.pretrained_model.fc = nn.Sequential(
        #                                 nn.Linear(self.embedding_size, self.tower_fc_units),
        #                                 nn.ReLU(inplace=True),
        # #                                 nn.Linear(self.tower_fc_units, self.tower_fc_units),
        # #                                 nn.ReLU(inplace=True),
        #                                 nn.Linear(self.tower_fc_units, 1),
        #                                 nn.Sigmoid()
        #                                 )

        if self.pretrained_model_arch == "resnet50":
            self.pretrained_model.fc = nn.Sequential(
                nn.ReLU(inplace=True),
            )
            self.sigmoid_tower = nn.Sequential(
                nn.Linear(self.embedding_size + self.num_covariates, self.tower_fc_units),
                nn.ReLU(inplace=True),
                # nn.Linear(self.tower_fc_units, self.tower_fc_units),
                # nn.ReLU(inplace=True),
                nn.Linear(self.tower_fc_units, 1),
                nn.Sigmoid()
            )
        elif self.pretrained_model_arch == "VGG16":
            self.pretrained_model.classifier[6] = nn.Sequential(

                nn.Linear(self.embedding_size, 1),
                nn.Linear(self.embedding_size, self.tower_fc_units),
                nn.ReLU(inplace=True),
                nn.Linear(self.embedding_size, self.tower_fc_units),
                nn.ReLU(inplace=True),
                nn.Linear((self.tower_fc_units), 1),
                nn.Sigmoid()
            )


    def forward(self, image_left, image_right, covariates):

        pretrained_features_left = self.pretrained_model(image_left)
        pretrained_features_right = self.pretrained_model(image_right)
        features_concat_covariates_left = torch.cat((pretrained_features_left, covariates), dim=1)
        features_concat_covariates_right = torch.cat((pretrained_features_right, covariates), dim=1)
        logit_left = self.sigmoid_tower(features_concat_covariates_left)
        logit_right = self.sigmoid_tower(features_concat_covariates_right)

        p = logit_right / (logit_left + logit_right)

        return p

class EmbeddingPredictionModelWithCovariates(nn.Module):

    def __init__(self,
                 tower_fc_units=12,
                 use_cuda=True,
                 use_covariates=True,
                 num_covariates=6):
        super(EmbeddingPredictionModelWithCovariates, self).__init__()

        self.tower_fc_units = tower_fc_units

        #self.classifier_tower = nn.Sequential(

            ##                         nn.Linear(self.embedding_size, 1),
            #nn.Linear(6+num_covariates, self.tower_fc_units),
            #nn.ReLU(inplace=True),
            #nn.Linear(self.tower_fc_units, 1),
            ##nn.Sigmoid()
        #)

        self.classifier_tower = nn.Sequential(

            #                         nn.Linear(self.embedding_size, 1),
            nn.Linear(6+num_covariates, self.tower_fc_units),
            nn.ReLU(inplace=True),
            nn.Linear(self.tower_fc_units, self.tower_fc_units),
            nn.ReLU(inplace=True),
            nn.Linear(self.tower_fc_units, 1),
            #nn.Sigmoid()
        )

        #self.classifier_tower = nn.Sequential(

            ##                         nn.Linear(self.embedding_size, 1),
            #nn.Linear(6+num_covariates, self.tower_fc_units*2),
            #nn.ReLU(inplace=True),
            #nn.Linear(self.tower_fc_units*2, self.tower_fc_units),
            #nn.ReLU(inplace=True),
            #nn.Linear(self.tower_fc_units, 1),
            ##nn.Sigmoid()
        #)


    def forward(self, embedding_left, embedding_right, covariates):

        embedding_right_minus_left = embedding_right - embedding_left
        #embedding_right_minus_left = embedding_left - embedding_right

        #print(embedding_right_minus_left.size())

        #embedding_concat_covariates_left = torch.cat((embedding_left, covariates), dim=1)
        #embedding_concat_covariates_right = torch.cat((embedding_right, covariates), dim=1)
        
        embedding_right_minus_left_concat_covariates = torch.cat([embedding_right_minus_left, covariates], dim=1)
        
        #print(embedding_right_minus_left_concat_covariates.size())

        #logit_left = self.classifier_tower(embedding_concat_covariates_left)
        #logit_right = self.classifier_tower(embedding_concat_covariates_right)

        # p = logit_right / (logit_left + logit_right)

        logit_right_minus_left = self.classifier_tower(embedding_right_minus_left_concat_covariates)

        #p = 1 / (1 + logit_right_minus_left)
        p = 1 / (1 + torch.exp(-logit_right_minus_left))

        #print(p)

        return p


class EmbeddingPredictionModelWithoutCovariates(nn.Module):

    def __init__(self,
                 tower_fc_units=12,
                 use_cuda=True):
        super(EmbeddingPredictionModelWithoutCovariates, self).__init__()

        self.tower_fc_units = tower_fc_units

        self.classifier_tower = nn.Sequential(

            #                         nn.Linear(self.embedding_size, 1),
            nn.Linear(6, self.tower_fc_units),
            nn.ReLU(inplace=True),
            nn.Linear(self.tower_fc_units, self.tower_fc_units),
            nn.ReLU(inplace=True),
            nn.Linear(self.tower_fc_units, 1),
            # nn.Sigmoid()
        )


    def forward(self, embedding_left, embedding_right):
        embedding_right_minus_left = embedding_right - embedding_left

        logit_right_minus_left = self.classifier_tower(embedding_right_minus_left)

        p = 1 / (1 + torch.exp(-logit_right_minus_left))

        # print(p)

        return p

# class EmbeddingPredictionModel(nn.Module):
#
#     def __init__(self,
#                  tower_fc_units=6,
#                  use_cuda=True):
#         super(EmbeddingPredictionModel, self).__init__()
#
#         self.tower_fc_units = tower_fc_units
#
#         self.classifier_tower = nn.Sequential(
#
#             #                         nn.Linear(self.embedding_size, 1),
#             nn.Linear(6, self.tower_fc_units),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.tower_fc_units, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, embedding_left, embedding_right):
#         logit_left = self.classifier_tower(embedding_left)
#         logit_right = self.classifier_tower(embedding_right)
#
#         p = logit_right / (logit_left + logit_right)
#
#         return p
