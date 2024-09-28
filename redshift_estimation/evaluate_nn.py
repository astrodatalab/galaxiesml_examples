import h5py
import tensorflow as tf
import numpy as np
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import tensorflow_probability as tfp
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
#import data set
import numpy as np
mnist = tf.keras.datasets.mnist
import random
from tensorboard.plugins.hparams import api as hp
import datetime
from tensorflow import keras
from sklearn.model_selection import train_test_split
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
from sklearn import ensemble

import random
from sklearn.model_selection import train_test_split
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
tfd = tfp.distributions
from photoz_utils import *

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
    
from tensorflow import keras
from sklearn.model_selection import train_test_split
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

tfd = tfp.distributions


from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

import h5py

import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load training data
photozdata_train = pd.read_hdf('photozdata_train_unscaled.h5', key='df')

# Load validation data
photozdata_val = pd.read_hdf('photozdata_val_unscaled.h5', key='df')

# Process training data
spectro_z_train = np.asarray(photozdata_train["specz"])
photodata_train = photozdata_train.drop("specz", axis=1)

# Process validation data
spectro_z_val = np.asarray(photozdata_val["specz"])
photodata_val = photozdata_val.drop("specz", axis=1)

# Cleaning
for data in [photodata_train, photodata_val]:
    data.replace(-99., np.nan, inplace=True)
    data.replace(-99.9, np.nan, inplace=True)
    data.replace(np.inf, np.nan, inplace=True)
    data.dropna(how='any', inplace=True)
   
photodata_train = photodata_train[['col1', 'col2', 'col3', 'col4', 'col5']]
photodata_val = photodata_val[['col1', 'col2', 'col3', 'col4', 'col5']]


    
# Initialize the scaler
min_max_scaler = MinMaxScaler()

# Fit the scaler only on the training data
min_max_scaler.fit(photodata_train)

# Transform both training and validation data using the fitted scaler
x_train = min_max_scaler.transform(photodata_train)
y_train = spectro_z_train

x_val = min_max_scaler.transform(photodata_val)
y_val = spectro_z_val

import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load training data
photozdata_test = pd.read_hdf('photozdata_test_unscaled.h5', key='df')
#photozdata_test = pd.read_hdf('photozdata_test_rescaled_no_duplicates.h5', key='df')


# Process training data
spectro_z_test = np.asarray(photozdata_test["specz"])
y_test = spectro_z_test
photodata_test = photozdata_test.drop("specz", axis=1)


# Cleaning
for data in [photodata_test]:
    data.replace(-99., np.nan, inplace=True)
    data.replace(-99.9, np.nan, inplace=True)
    data.replace(np.inf, np.nan, inplace=True)
    data.dropna(how='any', inplace=True)


photodata_test = photodata_test[['col1', 'col2', 'col3', 'col4', 'col5']]

# Fit the scaler only on the training data

min_max_scaler.fit(photodata_train)

x_test = min_max_scaler.transform(photodata_test)


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
    dz = delz(z_photo, z_spec)
    gamma = 0.15
    denominator = 1.0 + K.square(dz/gamma)
    L = 1 - 1.0 / denominator
    return L


def calculate_conv_outlier_rate2(z_photo,z_spec):

# This function calculate the conventional outlier rate. 

    outliers = []
    outliers_bayesian = []
    outlier_index_bayesian = []
    outlier_index = []
    for i in range(0,len(z_spec)):


        #outliers.append((abs(photoz[i] - y_test_original[i]))/(1+y_test_original[i]))
        outliers.append((abs(z_photo[i] - z_spec[i]))/(1+z_spec[i]))
        if outliers[i] > 0.15:
            outlier_index.append(i)
    
        outliers_bayesian.append((abs(z_photo[i] - z_spec[i]))/(1+z_spec[i]))
        if outliers_bayesian[i] > 0.15:
            outlier_index_bayesian.append(i)
            
    outlier_rate_conv = len(outlier_index_bayesian)/len(z_spec) 
    
    return outlier_rate_conv


def calculate_bayesian_outlier_rate2(z_photo,z_spec,zpdf_std):

# This function calculate the conventional outlier rate. 

    outliers = []
    outliers_bayesian = []
    outlier_index_bayesian = []
    for i in range(0,len(z_spec)):


        outliers_bayesian.append((abs(z_photo[i] - z_spec[i])-zpdf_std[i][0])/(1+z_spec[i]))
        if outliers_bayesian[i] > 0.15:
            outlier_index_bayesian.append(i)
            
    bayesian_outlier_rate = len(outlier_index_bayesian)/len(z_spec) 
    
    return bayesian_outlier_rate





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

def negative_loglikelihood(targets, estimated_distribution):
    return -estimated_distribution.log_prob(targets)

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([20,40,60,80,100]))
HP_NUM_EPOCHS = hp.HParam('num_epochs', hp.Discrete([500,1000]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd', 'RMSprop']))

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([40,60,80,120,160]))
HP_NUM_EPOCHS = hp.HParam('num_epochs', hp.Discrete([500,750,1000,1250,1500]))
HP_NUM_EPOCHS = hp.HParam('num_epochs', hp.Discrete([1000]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam']))

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([200]))

METRIC_ACCURACY = keras.metrics.RootMeanSquaredError()


checkpoint_path = 'evan_checkpoints/nn_save_99.ckpt'


cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 monitor='loss',
                                                 verbose=1,
                                                 save_freq = "epoch",
                                                 save_best_only=True)


params = {"lr": 0.0005, "epochs": 10000}


session_num = 0
coverage_list_mean = []
coverage_list_std = []
z_delta_div_uncertainty_list = []
z_delta_div_uncertainty = []
num_units = 200
hparams = {
  HP_NUM_UNITS: num_units
}
input_ = tf.keras.layers.Input(shape=x_train.shape[1:])
hidden1 = tf.keras.layers.Dense(200, activation="relu")(input_)
hidden2 = tf.keras.layers.Dense(200, activation="relu")(hidden1)
hidden3 = tf.keras.layers.Dense(200, activation="relu")(hidden2)
hidden4 = tf.keras.layers.Dense(200, activation="relu")(hidden3)

concat = tf.keras.layers.Concatenate()([input_, hidden4])
distribution_params = tf.keras.layers.Dense(units=1)(concat)
#output = tfp.layers.IndependentNormal(1)(distribution_params)
model = tf.keras.Model(inputs=[input_], outputs=[distribution_params])


optimizer = tf.keras.optimizers.Adam(learning_rate=params["lr"])
model.compile(optimizer=optimizer,  loss=calculate_loss,metrics=[keras.metrics.RootMeanSquaredError()])

 
# history = model.fit(x_train,y_train,epochs=params["epochs"],batch_size = 10000, shuffle = True,verbose=1,validation_data=(x_test,y_test), callbacks=[cp_callback])

model.load_weights('evan_checkpoints/nn_save_99.ckpt')
#model.load_weights('evan_checkpoints/cp7.ckpt')

#x_test = x_test.to_numpy()
#otherwise just continue:

evaluated_model = model(x_test)
#prediction_mean = (evaluated_model.mean()).numpy().tolist()


predictions = []
predictions = model.predict(x_test)
np.argmax(predictions[0])
photoz = []

for i in range(0,len(x_test)):
    photoz.append(predictions[i])
    
from photoz_utils import *

y_test_new = pd.Series(np.ravel(y_test))
photoz = pd.Series(np.ravel(photoz))
get_point_metrics(photoz,y_test_new)

data = {'photoz':photoz,
             'spectroz':y_test,
}


df = pd.DataFrame(data)
data = df

data.columns = ['photoz', 'spectroz']
data.to_csv('nn_save_99.csv')