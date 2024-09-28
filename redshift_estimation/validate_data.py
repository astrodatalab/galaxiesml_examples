import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
import keras
import os
import tensorboard
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorboard.plugins.hparams import api as hp

from photoz_utils import *
from DataMakerPlus import *

# examine the hdf5 files to make sure they are consistent

TRAIN_PATH = f'/data/HSC/HSC_v6/step2A/127x127/5x127x127_training_with_morphology.hdf5'
VALID_PATH = f'/data/HSC/HSC_v6/step2A/127x127/5x127x127_validation_with_morphology.hdf5'
TEST_PATH = f'/data/HSC/HSC_v6/step2A/127x127/5x127x127_testing_with_morphology.hdf5'

TRAIN_PATH64 = f'/data/HSC/HSC_v6/step2A/64x64/5x64x64_training_with_morphology.hdf5'
VALID_PATH64 = f'/data/HSC/HSC_v6/step2A/64x64/5x64x64_validation_with_morphology.hdf5'
TEST_PATH64 = f'/data/HSC/HSC_v6/step2A/64x64/5x64x64_testing_with_morphology.hdf5'

# load the HDF5 files and compare the object IDs between the 127x127 and 64x64 files

def load_hdf5(file_path):
    with h5py.File(file_path, 'r') as f:
        print(f'Keys: {list(f.keys())}')
        a_group_key = list(f.keys())[0]
        data = list(f[a_group_key])
        print(f'Length of data: {len(data)}')
        print(f'First 5 elements: {data[:5]}')
        return data
    

train_data = load_hdf5(TRAIN_PATH)
valid_data = load_hdf5(VALID_PATH)
test_data = load_hdf5(TEST_PATH)

train_data64 = load_hdf5(TRAIN_PATH64)
valid_data64 = load_hdf5(VALID_PATH64)
test_data64 = load_hdf5(TEST_PATH64)