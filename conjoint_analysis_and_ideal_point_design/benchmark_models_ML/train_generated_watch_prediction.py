"""
#############################################################################################
                Project: JMR23 - Disentangling Watches
                Author: Alex Burnap
                Email: alex.burnap@yale.edu
                Date: April 6, 2023
                License: MIT
#############################################################################################
"""
# Standard Libraries
import os
import math
import glob
import time

# Numerical Libraries
import numpy as np
import pandas as pd

# Image and Plotting Libraries
from PIL import Image
import matplotlib.pyplot as plt

# Neural Network Libraries
import torch
from torch import nn
import torch.nn.parallel
import torch.optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

# Local Libraries
from data_generated import WatchDataset, WatchDatasetEmbedded
from models import PretrainedBenchmarkModel, PretrainedWithCovariatesBenchmarkModel, EmbeddingPredictionModel

# -----------------------------------------------------------------------------------------------
#                    Globals
# -----------------------------------------------------------------------------------------------

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# -----------------------------------------------------------------------------------------------
#                    Real Watch Prediction Training Functions
# -----------------------------------------------------------------------------------------------

def train_pretrained_model(X_full_images=None,
                           Y_full=None,
                           # Z_full=None,
                           TRAIN_TASK_INDICES=None,
                           TEST_TASK_INDICES=None,
                           NUM_EPOCHS = 10,
                           BATCH_SIZE = 64,
                           arch = "resnet50",
                           learning_rate = 0.002
                           ):


    # Construct Model
    pretrained_model = PretrainedBenchmarkModel()
    pretrained_model.cuda()


    # Optimizers and Losses
    if arch == "resnet50":
        # ResNet50
        optimizer = torch.optim.Adam(pretrained_model.pretrained_model.fc.parameters(), lr=learning_rate)
    elif arch == "VGG":
        pass
        # optimizer = torch.optim.Adam(pretrained_model.pretrained_model.classifier.parameters(), lr=0.0002)
        # VGG19 - another layer
        # optimizer = torch.optim.Adam(pretrained_model.pretrained_model.classifier[6].parameters(), lr=0.0002)
    else:
        raise Exception("Architecture not found.")

    # Learning Rate Scheduler
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Losses
    # TODO: check BCELoss
    loss = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()

    # Data loading - Parallel
    train_dataset = WatchDataset(X_full_images=X_full_images[TRAIN_TASK_INDICES, :, :, :, :],
                                 Y_full=Y_full[TRAIN_TASK_INDICES])

    test_dataset = WatchDataset(X_full_images=X_full_images[TEST_TASK_INDICES, :, :, :, :],
                                Y_full=Y_full[TEST_TASK_INDICES])

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=4)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=BATCH_SIZE,
                                 shuffle=True,
                                 num_workers=4)


    data_loaders = {'train': train_dataloader, 'valid': test_dataloader}
    best_acc = 0.0
    dataset_sizes = {'train': len(data_loaders['train'].dataset),
                     'valid': len(data_loaders['valid'].dataset)}

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print("Epoch {}/{}".format(epoch, NUM_EPOCHS))
        for phase in ['train', 'valid']:
            if phase == 'train':
                #             exp_lr_scheduler.step()
                pretrained_model.train(True)
            else:
                pretrained_model.train(False)

            running_loss = 0.0
            running_corrects = 0

            for batch_ind, (images_left_batch, images_right_batch, y_batch) in enumerate(data_loaders[phase]):
                if batch_ind % 10 == 0:
                    print("Batch {}/{}".format(batch_ind, int(dataset_sizes[phase] / BATCH_SIZE)))


                optimizer.zero_grad()

                #             p_batch = pretrained_model(images_left_batch.cuda(), images_right_batch.cuda())
                p_batch = pretrained_model(Variable(images_left_batch.cuda()), Variable(images_right_batch.cuda())).squeeze()

                #             print(p_batch)

                predictions_batch = (p_batch > 0.5).float()
                loss = criterion(p_batch, y_batch.cuda())

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                #             running_loss += loss.data[0]
                running_loss += loss.item()
                running_corrects += torch.sum(predictions_batch == y_batch.cuda().data)

            if phase == 'train':
                train_epoch_loss = running_loss / dataset_sizes[phase]
                train_epoch_acc = running_corrects / dataset_sizes[phase]
            else:
                valid_epoch_loss = running_loss / dataset_sizes[phase]
                valid_epoch_acc = running_corrects / dataset_sizes[phase]

            if phase == 'valid' and valid_epoch_acc > best_acc:
                best_acc = valid_epoch_acc
                best_model_wts = pretrained_model.state_dict()

        print('Epoch [{}/{}] train loss: {:.4f} acc: {:.4f} '
              'valid loss: {:.4f} acc: {:.4f}'.format(
            epoch, NUM_EPOCHS - 1,
            train_epoch_loss, train_epoch_acc,
            valid_epoch_loss, valid_epoch_acc))

    print('Best val Acc: {:4f}'.format(best_acc))

    pretrained_model.load_state_dict(best_model_wts)
    return pretrained_model


def train_pretrained_model_with_covariates(X_full_images=None,
                                           Y_full=None,
                                           Z_full=None,
                                           TRAIN_TASK_INDICES=None,
                                           TEST_TASK_INDICES=None,
                                           NUM_EPOCHS = 10,
                                           BATCH_SIZE = 64,
                                           arch = "resnet50",
                                           learning_rate = 0.002
                                           ):


    # Construct Model
    pretrained_model = PretrainedWithCovariatesBenchmarkModel()
    pretrained_model.cuda()


    # Optimizers and Losses
    if arch == "resnet50":
        # ResNet50
        # optimizer = torch.optim.Adam(pretrained_model.pretrained_model.fc.parameters(), lr=learning_rate)
        optimizer = torch.optim.Adam(pretrained_model.sigmoid_tower.parameters(), lr=learning_rate)
    elif arch == "VGG":
        pass
        # optimizer = torch.optim.Adam(pretrained_model.pretrained_model.classifier.parameters(), lr=0.0002)
        # VGG19 - another layer
        # optimizer = torch.optim.Adam(pretrained_model.pretrained_model.classifier[6].parameters(), lr=0.0002)
    else:
        raise Exception("Architecture not found.")

    # Learning Rate Scheduler
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Losses
    # TODO: check BCELoss
    loss = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()

    # Data loading - Parallel
    train_dataset = WatchDataset(X_full_images=X_full_images[TRAIN_TASK_INDICES, :, :, :, :],
                                 Y_full=Y_full[TRAIN_TASK_INDICES],
                                 Z_full=Z_full[TRAIN_TASK_INDICES])

    test_dataset = WatchDataset(X_full_images=X_full_images[TEST_TASK_INDICES, :, :, :, :],
                                Y_full=Y_full[TEST_TASK_INDICES],
                                Z_full=Z_full[TEST_TASK_INDICES])

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=4)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=BATCH_SIZE,
                                 shuffle=True,
                                 num_workers=4)


    data_loaders = {'train': train_dataloader, 'valid': test_dataloader}
    best_acc = 0.0
    dataset_sizes = {'train': len(data_loaders['train'].dataset),
                     'valid': len(data_loaders['valid'].dataset)}

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print("Epoch {}/{}".format(epoch, NUM_EPOCHS))
        for phase in ['train', 'valid']:
            if phase == 'train':
                #             exp_lr_scheduler.step()
                pretrained_model.train(True)
            else:
                pretrained_model.train(False)

            running_loss = 0.0
            running_corrects = 0

            for batch_ind, (images_left_batch, images_right_batch, y_batch, z_batch) in enumerate(data_loaders[phase]):
                if batch_ind % 10 == 0:
                    print("Batch {}/{}".format(batch_ind, int(dataset_sizes[phase] / BATCH_SIZE)))


                optimizer.zero_grad()

                #             p_batch = pretrained_model(images_left_batch.cuda(), images_right_batch.cuda())
                p_batch = pretrained_model(Variable(images_left_batch.cuda()), Variable(images_right_batch.cuda()), Variable(z_batch.cuda())).squeeze()

                #             print(p_batch)

                predictions_batch = (p_batch > 0.5).float()
                loss = criterion(p_batch, y_batch.cuda())

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                #             running_loss += loss.data[0]
                running_loss += loss.item()
                running_corrects += torch.sum(predictions_batch == y_batch.cuda().data)

            if phase == 'train':
                train_epoch_loss = running_loss / dataset_sizes[phase]
                train_epoch_acc = running_corrects / dataset_sizes[phase]
            else:
                valid_epoch_loss = running_loss / dataset_sizes[phase]
                valid_epoch_acc = running_corrects / dataset_sizes[phase]

            if phase == 'valid' and valid_epoch_acc > best_acc:
                best_acc = valid_epoch_acc
                best_model_wts = pretrained_model.state_dict()

        print('Epoch [{}/{}] train loss: {:.4f} acc: {:.4f} '
              'valid loss: {:.4f} acc: {:.4f}'.format(
            epoch, NUM_EPOCHS - 1,
            train_epoch_loss, train_epoch_acc,
            valid_epoch_loss, valid_epoch_acc))

    print('Best val Acc: {:4f}'.format(best_acc))

    pretrained_model.load_state_dict(best_model_wts)
    return pretrained_model


def train_embedding_model(X_full_embeddings=None,
                          Y_full=None,
                          TRAIN_TASK_INDICES=None,
                          TEST_TASK_INDICES=None,
                          NUM_EPOCHS = 50,
                          BATCH_SIZE = 64,
                          learning_rate=0.0002):


    embedding_model = EmbeddingPredictionModel()
    embedding_model.cuda()

    # Data loading - Parallel
    train_dataset_embedding = WatchDatasetEmbedded(X_full_embeddings=X_full_embeddings[TRAIN_TASK_INDICES, :, :],
                                                   Y_full=Y_full[TRAIN_TASK_INDICES])

    test_dataset_embedding = WatchDatasetEmbedded(X_full_embeddings=X_full_embeddings[TEST_TASK_INDICES, :, :],
                                                  Y_full=Y_full[TEST_TASK_INDICES])

    train_dataloader_embedding = DataLoader(train_dataset_embedding,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                            num_workers=4)
    test_dataloader_embedding = DataLoader(test_dataset_embedding,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=4)

    # Optimizers

    # optimizer = torch.optim.Adam(pretrained_model_temp.pretrained_model.fc.parameters(), lr=0.001)
    # optimizer = torch.optim.Adam(pretrained_model_temp.pretrained_model.classifier.parameters(), lr=0.0002)
    optimizer = torch.optim.Adam(embedding_model.classifier_tower.parameters(), lr=learning_rate)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Losses
    # loss = nn.BCELoss()
    loss = nn.BCEWithLogitsLoss()
    criterion = torch.nn.CrossEntropyLoss()

    data_loaders = {'train': train_dataloader_embedding, 'valid': test_dataloader_embedding}
    best_acc = 0.0
    dataset_sizes = {'train': len(data_loaders['train'].dataset),
                     'valid': len(data_loaders['valid'].dataset)}

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print("Epoch {}/{}".format(epoch, NUM_EPOCHS))
        for phase in ['train', 'valid']:
            if phase == 'train':
                #             exp_lr_scheduler.step()
                embedding_model.train(True)
            else:
                embedding_model.train(False)

            running_loss = 0.0
            running_corrects = 0

            for batch_ind, (embeddings_left_batch, embeddings_right_batch, y_batch) in enumerate(data_loaders[phase]):
                if batch_ind % 10 == 0:
                    print("Batch {}/{}".format(batch_ind, int(dataset_sizes[phase] / BATCH_SIZE)))

                optimizer.zero_grad()

                #             p_batch = pretrained_model_temp(images_left_batch.cuda(), images_right_batch.cuda())
                p_batch = embedding_model(Variable(embeddings_left_batch.cuda()), Variable(embeddings_right_batch.cuda())).squeeze()

                #             print(p_batch)

                predictions_batch = (p_batch > 0.5).float()
                loss = criterion(p_batch, y_batch.cuda())
                #             loss = criterion(p_batch, y_batch.type(torch.LongTensor).cuda())
                #             loss = criterion(p_batch, y_batch.unsqueeze(1).type(torch.LongTensor) .cuda())
                #             loss = criterion(p_batch, y_batch.unsqueeze(1).cuda())
                #             print(loss)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                #             running_loss += loss.data[0]
                running_loss += loss.item()
                running_corrects += torch.sum(predictions_batch == y_batch.cuda().data)

            if phase == 'train':
                train_epoch_loss = running_loss / dataset_sizes[phase]
                train_epoch_acc = running_corrects / dataset_sizes[phase]
            else:
                valid_epoch_loss = running_loss / dataset_sizes[phase]
                valid_epoch_acc = running_corrects / dataset_sizes[phase]

            if phase == 'valid' and valid_epoch_acc > best_acc:
                best_acc = valid_epoch_acc
                best_model_wts = embedding_model.state_dict()

        print('Epoch [{}/{}] train loss: {:.4f} acc: {:.4f} '
              'valid loss: {:.4f} acc: {:.4f}'.format(
            epoch, NUM_EPOCHS - 1,
            train_epoch_loss, train_epoch_acc,
            valid_epoch_loss, valid_epoch_acc))

    print('Best val Acc: {:4f}'.format(best_acc))

    embedding_model.load_state_dict(best_model_wts)
    return embedding_model