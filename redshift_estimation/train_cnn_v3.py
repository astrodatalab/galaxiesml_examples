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
import argparse   

from photoz_utils import *
from DataMakerPlus import *

# Argument Parser for User Inputs
parser = argparse.ArgumentParser(description="Train CNN for redshift estimation.")
parser.add_argument('--image_size', type=int, default=64, choices=[64, 127],
                    help="Image size to use: 64 or 127 (default: 64).")
parser.add_argument('--epochs', type=int, default=200, 
                    help="Number of training epochs (default: 200).")
parser.add_argument('--batch_size', type=int, default=256, 
                    help="Batch size for training (default: 256).")
parser.add_argument('--learning_rate', type=float, default=0.0001, 
                    help="Learning rate (default: 0.0001).")
args = parser.parse_args()


# name of the model to save the models
model_name = 'HSC_v6_CNN_delta_v2_galaxiesml_script'

# set the directories for the model and logs
checkpoint_filepath = f'/data2/models/{model_name}/checkpoints/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_filepath)
log_dir = os.path.join('/data2/logs/', model_name)

# allocate 15 GB of GPU memory
GB_LIMIT = 15

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(GB_LIMIT*1000)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
        
# User-Selected Image Size
# Dynamically set based on user input
image_size = args.image_size
image_size_str = str(image_size)  # Convert to string for logging

# Hyperparameters
# Set hyperparameters for training
IMAGE_SHAPE = (5, image_size, image_size)  # 5 color channels  
NUM_DENSE_UNITS = 200
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
Z_MAX = 4

# Store hyperparameters in a dictionary for easy reference
hparams = {
    'num_dense_units': NUM_DENSE_UNITS,
    'batch_size': BATCH_SIZE,
    'num_epochs': NUM_EPOCHS,
    'learning_rate': LEARNING_RATE,
    'z_max': Z_MAX
}


# Dataset Paths
# Set paths for training, validation, and testing datasets
TRAIN_PATH = "E:/Datasets/5x64x64_training_with_morphology.hdf5"
VAL_PATH = "E:/Datasets/5x64x64_validation_with_morphology.hdf5"
TEST_PATH = "E:/Datasets/5x64x64_testing_with_morphology.hdf5"

# Check Dataset Paths
# Ensure datasets exist before proceeding
for path in [TRAIN_PATH, VAL_PATH, TEST_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

# Load the number of training samples
with h5py.File(TRAIN_PATH, 'r') as f:
    train_len = len(f['specz_redshift'])
print(train_len)

# List parameter names based on dataset structure
param_names = []
for i in ['g', 'r', 'i', 'z', 'y']:
    for j in ['cmodel_mag']:
        param_names.append(i + '_' + j)
        
# Generator Arguments
gen_args = {
    'image_key': 'image',
    'numerical_keys': param_names,
    'y_key': 'specz_redshift',
    'scaler': True,
    'labels_encoding': False,
    'batch_size': hparams['batch_size'],
    'shuffle': False
}

# Create data generators for training, validation, and testing
train_gen = HDF5DataGenerator(TRAIN_PATH, mode='train', **gen_args)
val_gen = HDF5DataGenerator(VAL_PATH, mode='train', **gen_args)
test_gen = HDF5DataGenerator(TEST_PATH, mode='test', **gen_args)

# Define posterior and prior trainable functions for Bayesian layers
import tensorflow_probability as tfp
tfd = tfp.distributions
def posterior_mean_field(kernel_size: int, bias_size: int, dtype: any) -> tf.keras.Model:
    """Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`."""
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))

    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype, initializer=lambda shape, dtype: random_gaussian_initializer(shape, dtype), trainable=True),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :n],
                       scale= + 10e-4*tf.nn.softplus(c + t[..., n:])),
            reinterpreted_batch_ndims=1)),
    ])
def prior_trainable(kernel_size: int, bias_size: int, dtype: any) -> tf.keras.Model:
    """Specify the prior over `keras.layers.Dense` `kernel` and `bias`."""
    n = kernel_size + bias_size

    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),  # Returns a trainable variable of shape n, regardless of input
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t, scale=1),
            reinterpreted_batch_ndims=1)),
    ])
def random_gaussian_initializer(shape, dtype):
    """Random initializer for Gaussian distribution."""
    n = int(shape / 2)
    loc_norm = tf.random_normal_initializer(mean=0., stddev=0.1)
    loc = tf.Variable(
        initial_value=loc_norm(shape=(n,), dtype=dtype)
    )
    scale_norm = tf.random_normal_initializer(mean=-3., stddev=0.1)
    scale = tf.Variable(
        initial_value=scale_norm(shape=(n,), dtype=dtype)
    )
    return tf.concat([loc, scale], 0)

# Loss function for the model
def negative_loglikelihood(targets, estimated_distribution):
    return -estimated_distribution.log_prob(targets)

# Model Definition
# Define CNN and dense neural network
input_cnn = Input(shape=IMAGE_SHAPE)  # Input layer for image data
input_nn = Input(shape=(5,))  # Input layer for numerical data

# CNN
# Convolutional layers process the image input
conv1 = Conv2D(32, kernel_size=(3, 3), activation='tanh', padding='same', data_format='channels_first')(input_cnn)
pool1 = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(conv1)
conv2 = Conv2D(64, kernel_size=(3, 3), activation='tanh', padding='same', data_format='channels_first')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(conv2)
flatten = Flatten()(pool2)  # Flatten the features for the dense layers

# Dense layers for image features
dense1 = Dense(512, activation='tanh')(flatten)
dense2 = Dense(128, activation='tanh')(dense1)
dense3 = Dense(32, activation='tanh')(dense2)

# NN
# Dense layers process the numerical inputs
hidden1 = Dense(hparams['num_dense_units'], activation="relu")(input_nn)
hidden2 = Dense(hparams['num_dense_units'], activation="relu")(hidden1)

# Concat & Output
# Combine features from CNN and NN and pass through final dense layer
concat = Concatenate()([dense3, hidden2])
output = Dense(1)(concat)

# Define the model
model = Model(inputs=[input_cnn, input_nn], outputs=[output])


# Custom Loss Function
import keras.backend as K
def calculate_loss(z_photo, z_spec):
    """
    HSC METRIC. Returns an array. Loss is accuracy metric defined by HSC, meant
    to capture the effects of bias, scatter, and outlier all in one. This has
    uses for both point and density estimation.
    z_photo: array
        Photometric or predicted redshifts.
    z_spec: array
        Spectroscopic or actual redshifts.
    """
    dz = z_photo-z_spec
    gamma = 0.15
    denominator = 1.0 + K.square(dz/gamma)
    L = 1 - 1.0 / denominator
    return L

# Compile Model
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=calculate_loss, metrics=[tf.keras.metrics.RootMeanSquaredError()])

# Callbacks for Model Training
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_freq='epoch',
    save_best_only=True,
    verbose=True)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
hparam_callback = hp.KerasCallback(log_dir, hparams)

# Train the Model
model.fit(train_gen, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, shuffle=True, verbose=1, validation_data=val_gen, callbacks=[tensorboard_callback, model_checkpoint_callback, hparam_callback])

# model.load_weights(checkpoint_filepath)

# pred = model.predict(test_gen)

# with h5py.File(TEST_PATH, 'r') as file:
#     y_test = np.asarray(file['specz_redshift'][:])
#     oid_test = np.asarray(file['object_id'][:])
    
# plot_predictions(np.ravel(pred), y_test)

# metrics = get_point_metrics(pd.Series(np.ravel(pred)), pd.Series(y_test), binned=False)

# print(metrics)