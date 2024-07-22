"""
Project: Generative Interpretable Visual Design: Using Disentanglement for Visual Conjoint Analysis
Author: Ankit Sisodia, Alex Burnap, Vineet Kumar
Email: asisodia@purdue.edu, alex.burnap@yale.edu, vineet.kumar@yale.edu
Date: July 2024
License: MIT
"""
import subprocess
import os
import abc
import hashlib
import zipfile
import glob
import logging
import tarfile
from skimage.io import imread
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from shutil import copyfile

DIR = os.path.abspath(os.path.dirname(__file__))
COLOUR_BLACK = 0
COLOUR_WHITE = 1

def get_img_size(dataset):
    """Return the correct image size."""
    return (3, 128, 128) 

def get_background(dataset):
    """Return the image background color."""
    return COLOUR_WHITE

def split_dataset(model_name):
    """Split the dataset."""

    root=os.path.join(DIR, '../data/watches/') 
    data = os.path.join(root, "christies.npz")

    dataset_zip = np.load(data)
    watches = dataset_zip['watches']
 
    continuousprice = dataset_zip['continuousprice']
    discreteprice = dataset_zip['discreteprice']
    brand = dataset_zip['brand']
    circa = dataset_zip['circa']
    material = dataset_zip['material']
    movement = dataset_zip['movement']
    discreteprice_brand = dataset_zip['discreteprice_brand']
    discreteprice_circa = dataset_zip['discreteprice_circa']
    discreteprice_material = dataset_zip['discreteprice_material']
    discreteprice_movement = dataset_zip['discreteprice_movement']
    brand_circa = dataset_zip['brand_circa']
    brand_material = dataset_zip['brand_material']
    brand_movement = dataset_zip['brand_movement']
    circa_material = dataset_zip['circa_material']
    circa_movement = dataset_zip['circa_movement']
    material_movement = dataset_zip['material_movement']
    discreteprice_brand_circa = dataset_zip['discreteprice_brand_circa']
    discreteprice_brand_material = dataset_zip['discreteprice_brand_material']
    discreteprice_brand_movement = dataset_zip['discreteprice_brand_movement']
    discreteprice_circa_material = dataset_zip['discreteprice_circa_material']
    discreteprice_circa_movement = dataset_zip['discreteprice_circa_movement']
    discreteprice_material_movement = dataset_zip['discreteprice_material_movement']
    brand_circa_material = dataset_zip['brand_circa_material']
    brand_circa_movement = dataset_zip['brand_circa_movement']
    brand_material_movement = dataset_zip['brand_material_movement']
    circa_material_movement = dataset_zip['circa_material_movement']
    discreteprice_brand_circa_material = dataset_zip['discreteprice_brand_circa_material']
    discreteprice_brand_circa_movement = dataset_zip['discreteprice_brand_circa_movement']
    discreteprice_brand_material_movement = dataset_zip['discreteprice_brand_material_movement']
    discreteprice_circa_material_movement = dataset_zip['discreteprice_circa_material_movement']
    brand_circa_material_movement = dataset_zip['brand_circa_material_movement']
    discreteprice_brand_circa_material_movement = dataset_zip['discreteprice_brand_circa_material_movement']

    modelname = dataset_zip['modelname']
    filenames = dataset_zip['filenames']
   
    sequence = np.arange(0,filenames.shape[0])
    df = pd.DataFrame(data=np.column_stack((sequence,modelname,filenames)),columns=['seq','model','file'])
    df['model'] = df['model'].str.encode('ascii', 'ignore').str.decode('ascii')

    df_mod = df.groupby(['model'])["seq"].count().reset_index(name="count")
    r = np.random.uniform(size=df_mod.shape[0])
    r = np.where(r>=0.9,1,0) ## split ratio
    df_mod['r'] = r.tolist()
    result = pd.merge(df, df_mod, on="model")
    train_idx = result[result['r']==0]
    valid_idx = result[result['r']==1]
    train_idx = train_idx['seq'].to_numpy()
    valid_idx = valid_idx['seq'].to_numpy()
    train_idx = train_idx.astype(np.int)
    valid_idx = valid_idx.astype(np.int)
    np.savez( os.path.join(DIR, "../results",model_name,"christies_train.npz"),watches=watches[train_idx,:,:,],continuousprice=continuousprice[train_idx],discreteprice=discreteprice[train_idx,],brand=brand[train_idx,],circa=circa[train_idx,],material=material[train_idx,],movement=movement[train_idx,],discreteprice_brand=discreteprice_brand[train_idx,],discreteprice_circa=discreteprice_circa[train_idx,],discreteprice_material=discreteprice_material[train_idx,],discreteprice_movement=discreteprice_movement[train_idx,],brand_circa=brand_circa[train_idx,],brand_material=brand_material[train_idx,],brand_movement=brand_movement[train_idx,],circa_material=circa_material[train_idx,],circa_movement=circa_movement[train_idx,],material_movement=material_movement[train_idx,],discreteprice_brand_circa=discreteprice_brand_circa[train_idx,],discreteprice_brand_material=discreteprice_brand_material[train_idx,],discreteprice_brand_movement=discreteprice_brand_movement[train_idx,],discreteprice_circa_material=discreteprice_circa_material[train_idx,],discreteprice_circa_movement=discreteprice_circa_movement[train_idx,],discreteprice_material_movement=discreteprice_material_movement[train_idx,],brand_circa_material=brand_circa_material[train_idx,],brand_circa_movement=brand_circa_movement[train_idx,],brand_material_movement=brand_material_movement[train_idx,],circa_material_movement=circa_material_movement[train_idx,],discreteprice_brand_circa_material=discreteprice_brand_circa_material[train_idx,],discreteprice_brand_circa_movement=discreteprice_brand_circa_movement[train_idx,],discreteprice_brand_material_movement=discreteprice_brand_material_movement[train_idx,],discreteprice_circa_material_movement=discreteprice_circa_material_movement[train_idx,],brand_circa_material_movement=brand_circa_material_movement[train_idx,],discreteprice_brand_circa_material_movement=discreteprice_brand_circa_material_movement[train_idx,],filenames=filenames[train_idx])
    np.savez( os.path.join(DIR, "../results",model_name,"christies_validation.npz"),watches=watches[valid_idx,:,:,],continuousprice=continuousprice[valid_idx],discreteprice=discreteprice[valid_idx,],brand=brand[valid_idx,],circa=circa[valid_idx,],material=material[valid_idx,],movement=movement[valid_idx,],discreteprice_brand=discreteprice_brand[valid_idx,],discreteprice_circa=discreteprice_circa[valid_idx,],discreteprice_material=discreteprice_material[valid_idx,],discreteprice_movement=discreteprice_movement[valid_idx,],brand_circa=brand_circa[valid_idx,],brand_material=brand_material[valid_idx,],brand_movement=brand_movement[valid_idx,],circa_material=circa_material[valid_idx,],circa_movement=circa_movement[valid_idx,],material_movement=material_movement[valid_idx,],discreteprice_brand_circa=discreteprice_brand_circa[valid_idx,],discreteprice_brand_material=discreteprice_brand_material[valid_idx,],discreteprice_brand_movement=discreteprice_brand_movement[valid_idx,],discreteprice_circa_material=discreteprice_circa_material[valid_idx,],discreteprice_circa_movement=discreteprice_circa_movement[valid_idx,],discreteprice_material_movement=discreteprice_material_movement[valid_idx,],brand_circa_material=brand_circa_material[valid_idx,],brand_circa_movement=brand_circa_movement[valid_idx,],brand_material_movement=brand_material_movement[valid_idx,],circa_material_movement=circa_material_movement[valid_idx,],discreteprice_brand_circa_material=discreteprice_brand_circa_material[valid_idx,],discreteprice_brand_circa_movement=discreteprice_brand_circa_movement[valid_idx,],discreteprice_brand_material_movement=discreteprice_brand_material_movement[valid_idx,],discreteprice_circa_material_movement=discreteprice_circa_material_movement[valid_idx,],brand_circa_material_movement=brand_circa_material_movement[valid_idx,],discreteprice_brand_circa_material_movement=discreteprice_brand_circa_material_movement[valid_idx,],filenames=filenames[valid_idx])

    copyfile(os.path.join(root, "christies_test1.npz"),os.path.join(DIR, "../results",model_name,"christies_test1.npz"))
    copyfile(os.path.join(root, "christies_test2.npz"),os.path.join(DIR, "../results",model_name,"christies_test2.npz"))

    return 0

def get_dataloaders(dataset, root=None, shuffle=True, pin_memory=True,
                    batch_size=128,eval_batchsize=10000,model_name="temp",sup_signal="brand",logger=logging.getLogger(__name__), **kwargs):
    """A generic data loader
    Parameters
    ----------
    dataset :   Name of the dataset to load
    root : str  Path to the dataset root. If `None` uses the default one.
    kwargs :    Additional arguments to `DataLoader`. Default values are modified.
    """
    pin_memory = pin_memory and torch.cuda.is_available  # only pin if GPU available

    temp = split_dataset(model_name)

    Train_Dataset = Watches(split="train",model_name=model_name,sup_signal=sup_signal,logger=logger)
    Validation_Dataset = Watches(split="validation",model_name=model_name,sup_signal=sup_signal,logger=logger)
    Test1_Dataset = Watches(split="test1",model_name=model_name,sup_signal=sup_signal,logger=logger)
    Test2_Dataset = Watches(split="test2",model_name=model_name,sup_signal=sup_signal,logger=logger)

    train_loader = DataLoader(Train_Dataset,batch_size=batch_size,shuffle=True,pin_memory=pin_memory,**kwargs)
    validation_loader = DataLoader(Validation_Dataset,batch_size=eval_batchsize,shuffle=True,pin_memory=pin_memory,**kwargs)
    test1_loader = DataLoader(Test1_Dataset,batch_size=eval_batchsize,shuffle=False,pin_memory=pin_memory,**kwargs)
    test2_loader = DataLoader(Test2_Dataset,batch_size=eval_batchsize,shuffle=False,pin_memory=pin_memory,**kwargs)
    train_loader_all = DataLoader(Train_Dataset,batch_size=eval_batchsize,shuffle=False,pin_memory=pin_memory,**kwargs) 
    train_loader_one = DataLoader(Train_Dataset,batch_size=1,shuffle=True,pin_memory=pin_memory,**kwargs)
    
    return train_loader, validation_loader, test1_loader, test2_loader, train_loader_all, train_loader_one 

class Watches(Dataset):
    """
    """
    files = {"train": "christies_train.npz", "validation": "christies_validation.npz", "test1": "christies_test1.npz", "test2": "christies_test2.npz", "all":"christies.npz"} 
    img_size = (3, 128, 128)
    background_color = COLOUR_WHITE
    def __init__(self, root=os.path.join(DIR, '../results/'), transforms_list=[transforms.ToTensor()], logger=logging.getLogger(__name__), split="train",model_name="temp",sup_signal="brand",**kwargs):
        self.model_name = model_name
        self.sup_signal = sup_signal
        self.data = os.path.join(DIR,'../results/',self.model_name, type(self).files[split])

        self.transforms = transforms.Compose(transforms_list)
        self.logger = logger

        dataset_zip = np.load(self.data)
        self.imgs = dataset_zip['watches']
        self.brand = dataset_zip['brand']
        self.circa = dataset_zip['circa']
        self.material = dataset_zip['material']
        self.movement = dataset_zip['movement']
        self.continuousprice = dataset_zip['continuousprice']
        self.continuousprice=(self.continuousprice-np.mean(self.continuousprice))/np.std(self.continuousprice)
        self.discreteprice = dataset_zip['discreteprice']
        self.discreteprice_brand = dataset_zip['discreteprice_brand']
        self.discreteprice_circa = dataset_zip['discreteprice_circa']
        self.discreteprice_material = dataset_zip['discreteprice_material']
        self.discreteprice_movement = dataset_zip['discreteprice_movement']
        self.brand_circa = dataset_zip['brand_circa']
        self.brand_material = dataset_zip['brand_material']
        self.brand_movement = dataset_zip['brand_movement']
        self.circa_material = dataset_zip['circa_material']
        self.circa_movement = dataset_zip['circa_movement']
        self.material_movement = dataset_zip['material_movement']
        self.discreteprice_brand_circa = dataset_zip['discreteprice_brand_circa']
        self.discreteprice_brand_material = dataset_zip['discreteprice_brand_material']
        self.discreteprice_brand_movement = dataset_zip['discreteprice_brand_movement']
        self.discreteprice_circa_material = dataset_zip['discreteprice_circa_material']
        self.discreteprice_circa_movement = dataset_zip['discreteprice_circa_movement']
        self.discreteprice_material_movement = dataset_zip['discreteprice_material_movement']
        self.brand_circa_material = dataset_zip['brand_circa_material']
        self.brand_circa_movement = dataset_zip['brand_circa_movement']
        self.brand_material_movement = dataset_zip['brand_material_movement']
        self.circa_material_movement = dataset_zip['circa_material_movement']
        self.discreteprice_brand_circa_material = dataset_zip['discreteprice_brand_circa_material']
        self.discreteprice_brand_circa_movement = dataset_zip['discreteprice_brand_circa_movement']
        self.discreteprice_brand_material_movement = dataset_zip['discreteprice_brand_material_movement']
        self.discreteprice_circa_material_movement = dataset_zip['discreteprice_circa_material_movement']
        self.brand_circa_material_movement = dataset_zip['brand_circa_material_movement']
        self.discreteprice_brand_circa_material_movement = dataset_zip['discreteprice_brand_circa_material_movement']
        self.filenames = dataset_zip['filenames']

        if self.sup_signal == 'brand':
           self.signal_value = self.brand
           self.signal_value = np.argmax(self.signal_value,axis=1)
        elif self.sup_signal == 'circa':
           self.signal_value = self.circa
           self.signal_value = np.argmax(self.signal_value,axis=1)
        elif self.sup_signal == 'material':
           self.signal_value = self.material
           self.signal_value = np.argmax(self.signal_value,axis=1)
        elif self.sup_signal == 'movement':
           self.signal_value = self.movement
           self.signal_value = np.argmax(self.signal_value,axis=1)
        elif self.sup_signal == 'discreteprice':
           self.signal_value = self.discreteprice
           self.signal_value = np.argmax(self.signal_value,axis=1)
        elif self.sup_signal == 'discreteprice_brand':
           self.signal_value = self.discreteprice_brand
           self.signal_value = np.argmax(self.signal_value,axis=1)
        elif self.sup_signal == 'discreteprice_circa':
           self.signal_value = self.discreteprice_circa
           self.signal_value = np.argmax(self.signal_value,axis=1)
        elif self.sup_signal == 'discreteprice_material':
            self.signal_value = self.discreteprice_material
            self.signal_value = np.argmax(self.signal_value,axis=1)
        elif self.sup_signal == 'discreteprice_movement':
            self.signal_value = self.discreteprice_movement
            self.signal_value = np.argmax(self.signal_value,axis=1)
        elif self.sup_signal == 'brand_circa':
            self.signal_value = self.brand_circa
            self.signal_value = np.argmax(self.signal_value,axis=1)
        elif self.sup_signal == 'brand_material':
            self.signal_value = self.brand_material
            self.signal_value = np.argmax(self.signal_value,axis=1)
        elif self.sup_signal == 'brand_movement':
            self.signal_value = self.brand_movement
            self.signal_value = np.argmax(self.signal_value,axis=1)
        elif self.sup_signal == 'circa_material':
            self.signal_value = self.circa_material
            self.signal_value = np.argmax(self.signal_value,axis=1)
        elif self.sup_signal == 'circa_movement':
            self.signal_value = self.circa_movement
            self.signal_value = np.argmax(self.signal_value,axis=1)
        elif self.sup_signal == 'material_movement':
            self.signal_value = self.material_movement
            self.signal_value = np.argmax(self.signal_value,axis=1)
        elif self.sup_signal == 'discreteprice_brand_circa':
            self.signal_value = self.discreteprice_brand_circa
            self.signal_value = np.argmax(self.signal_value,axis=1)
        elif self.sup_signal == 'discreteprice_brand_material':
            self.signal_value = self.discreteprice_brand_material
            self.signal_value = np.argmax(self.signal_value,axis=1)
        elif self.sup_signal == 'discreteprice_brand_movement':
            self.signal_value = self.discreteprice_brand_movement
            self.signal_value = np.argmax(self.signal_value,axis=1)
        elif self.sup_signal == 'discreteprice_circa_material':
            self.signal_value = self.discreteprice_circa_material
            self.signal_value = np.argmax(self.signal_value,axis=1)
        elif self.sup_signal == 'discreteprice_circa_movement':
            self.signal_value = self.discreteprice_circa_movement
            self.signal_value = np.argmax(self.signal_value,axis=1)
        elif self.sup_signal == 'discreteprice_material_movement':
            self.signal_value = self.discreteprice_material_movement
            self.signal_value = np.argmax(self.signal_value,axis=1)
        elif self.sup_signal == 'brand_circa_material':
            self.signal_value = self.brand_circa_material
            self.signal_value = np.argmax(self.signal_value,axis=1)
        elif self.sup_signal == 'brand_circa_movement':
            self.signal_value = self.brand_circa_movement
            self.signal_value = np.argmax(self.signal_value,axis=1)
        elif self.sup_signal == 'brand_material_movement':
            self.signal_value = self.brand_material_movement
            self.signal_value = np.argmax(self.signal_value,axis=1)
        elif self.sup_signal == 'circa_material_movement':
            self.signal_value = self.circa_material_movement
            self.signal_value = np.argmax(self.signal_value,axis=1)
        elif self.sup_signal == 'discreteprice_brand_circa_material':
            self.signal_value = self.discreteprice_brand_circa_material
            self.signal_value = np.argmax(self.signal_value,axis=1)
        elif self.sup_signal == 'discreteprice_brand_circa_movement':
            self.signal_value = self.discreteprice_brand_circa_movement
            self.signal_value = np.argmax(self.signal_value,axis=1)
        elif self.sup_signal == 'discreteprice_brand_material_movement':
            self.signal_value = self.discreteprice_brand_material_movement
            self.signal_value = np.argmax(self.signal_value,axis=1)
        elif self.sup_signal == 'discreteprice_circa_material_movement':
            self.signal_value = self.discreteprice_circa_material_movement
            self.signal_value = np.argmax(self.signal_value,axis=1)
        elif self.sup_signal == 'brand_circa_material_movement':
            self.signal_value = self.brand_circa_material_movement
            self.signal_value = np.argmax(self.signal_value,axis=1)
        elif self.sup_signal == 'discreteprice_brand_circa_material_movement':
            self.signal_value = self.discreteprice_brand_circa_material_movement
            self.signal_value = np.argmax(self.signal_value,axis=1)
        elif self.sup_signal == 'continuousprice':
           self.signal_value = dataset_zip['continuousprice']
           self.signal_value = (self.signal_value-np.mean(self.signal_value))/np.std(self.signal_value)

    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        """Get the image of `idx`
        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        lat_value : np.array
            Array of length 6, that gives the value of each factor of variation.
        """
        img = self.transforms(self.imgs[idx])
        continuousprice = self.continuousprice[idx]
        discreteprice = self.discreteprice[idx]
        brand = self.brand[idx]
        circa = self.circa[idx]
        material = self.material[idx]
        movement = self.movement[idx]
        discreteprice_brand = self.discreteprice_brand[idx]
        discreteprice_circa = self.discreteprice_circa[idx]
        discreteprice_material = self.discreteprice_material[idx]
        discreteprice_movement = self.discreteprice_movement[idx]
        brand_circa = self.brand_circa[idx]
        brand_material = self.brand_material[idx]
        brand_movement = self.brand_movement[idx]
        circa_material = self.circa_material[idx]
        circa_movement = self.circa_movement[idx]
        material_movement = self.material_movement[idx]
        discreteprice_brand_circa = self.discreteprice_brand_circa[idx]
        discreteprice_brand_material = self.discreteprice_brand_material[idx]
        discreteprice_brand_movement = self.discreteprice_brand_movement[idx]
        discreteprice_circa_material = self.discreteprice_circa_material[idx]
        discreteprice_circa_movement = self.discreteprice_circa_movement[idx]
        discreteprice_material_movement = self.discreteprice_material_movement[idx]
        brand_circa_material = self.brand_circa_material[idx]
        brand_circa_movement = self.brand_circa_movement[idx]
        brand_material_movement = self.brand_material_movement[idx]
        circa_material_movement = self.circa_material_movement[idx]
        discreteprice_brand_circa_material = self.discreteprice_brand_circa_material[idx]
        discreteprice_brand_circa_movement = self.discreteprice_brand_circa_movement[idx]
        discreteprice_brand_material_movement = self.discreteprice_brand_material_movement[idx]
        discreteprice_circa_material_movement = self.discreteprice_circa_material_movement[idx]
        brand_circa_material_movement = self.brand_circa_material_movement[idx]
        discreteprice_brand_circa_material_movement = self.discreteprice_brand_circa_material_movement[idx]
        signal_value = self.signal_value[idx]

        filenames = self.filenames[idx]

        return img, 0, signal_value, continuousprice, discreteprice, brand, circa, material, movement, discreteprice_brand, discreteprice_circa, discreteprice_material, discreteprice_movement, brand_circa, brand_material, brand_movement, circa_material, circa_movement, material_movement, discreteprice_brand_circa, discreteprice_brand_material, discreteprice_brand_movement, discreteprice_circa_material, discreteprice_circa_movement, discreteprice_material_movement, brand_circa_material, brand_circa_movement, brand_material_movement, circa_material_movement, discreteprice_brand_circa_material, discreteprice_brand_circa_movement, discreteprice_brand_material_movement, discreteprice_circa_material_movement, brand_circa_material_movement, discreteprice_brand_circa_material_movement, filenames

# HELPERS
def preprocess(root, size=(128, 128), img_format='JPEG', center_crop=None):
    """Preprocess a folder of images.

    Parameters
    ----------
    root : string
        Root directory of all images.

    size : tuple of int
        Size (width, height) to rescale the images. If `None` don't rescale.

    img_format : string
        Format to save the image in. Possible formats:
        https://pillow.readthedocs.io/en/3.1.x/handbook/image-file-formats.html.

    center_crop : tuple of int
        Size (width, height) to center-crop the images. If `None` don't center-crop.
    """
    imgs = []
    for ext in [".png", ".jpg", ".jpeg"]:
        imgs += glob.glob(os.path.join(root, '*' + ext))

    for img_path in tqdm(imgs):
        img = Image.open(img_path)
        width, height = img.size

        if size is not None and width != size[1] or height != size[0]:
            img = img.resize(size, Image.ANTIALIAS)

        if center_crop is not None:
            new_width, new_height = center_crop
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = (width + new_width) // 2
            bottom = (height + new_height) // 2

            img.crop((left, top, right, bottom))

        img.save(img_path, img_format)
