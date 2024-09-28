#goal: provide basic implementation of deep learning NN for general application
#goal of this version is to set up a gridsearch with additional params (optimizers, loss functions)

#goal: provide basic implementation of deep learning NN for general application
import tensorflow as tf
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
from sklearn import linear_model
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
#import data set
import numpy as np
mnist = tf.keras.datasets.mnist
import random
from sklearn.model_selection import train_test_split
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
tfd = tfp.distributions
#import photoz data:
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("GPUs Available: ", len(physical_devices))
#if physical_devices:
#   tf.config.experimental.set_memory_growth(physical_devices[0], True)

import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorboard.plugins.hparams import api as hp
import pandas as pd
#import data set
import numpy as np
mnist = tf.keras.datasets.mnist
import random
import datetime
from tensorflow import keras
from sklearn.model_selection import train_test_split
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#import photoz data:
tfd = tfp.distributions
#from google.colab import files

photozdata = pd.read_csv('/data/HSC/HSC_IMAGES_FIXED/HSC_photozdata_full_header_trimmed.csv')
spectro_z = np.asarray(photozdata["specz_redshift"])


col1 = np.asarray(photozdata["g_cmodel_mag"])
col2 = np.asarray(photozdata["r_cmodel_mag"])
col3 = np.asarray(photozdata["i_cmodel_mag"])
col4 =np.asarray(photozdata["z_cmodel_mag"])
col5 = np.asarray(photozdata["y_cmodel_mag"])

#scaled columns between 0 and 1
col1 /= np.max(np.abs(col1),axis=0)
col2 /= np.max(np.abs(col2),axis=0)
col3 /= np.max(np.abs(col3),axis=0)
col4 /= np.max(np.abs(col4),axis=0)
col5 /= np.max(np.abs(col5),axis=0)
#photodata = np.column_stack((col1,col2,col3,col4,col5))

photodata = {'col1':col1,
             'col2':col2,
             'col3':col3,
             'col4':col4,
             'col5':col5,
}

df = pd.DataFrame(photodata)
photodata = df

spectro_z = pd.DataFrame(spectro_z)

#bin the redshift:
bin_size = 0.1
#bin_size = 0.12
#bin the redshift:
bin_size = 0.1


x_train ,x_test,y_train,y_test = train_test_split(photodata,spectro_z,test_size=0.5)

y_train = np.array(y_train)
y_train = np.round(y_train/bin_size)
y_test = np.array(y_test)
y_test_original = y_test
y_test = np.round(y_test/bin_size)


x_train = np.array(x_train)

x_test = np.array(x_test)

#bin_size = 0.12
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
def prior_trainable(kernel_size: int, bias_size: int, dtype: any) -> tf.keras.Model:
    """Specify the prior over `keras.layers.Dense` `kernel` and `bias`."""
    n = kernel_size + bias_size

    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),  # Returns a trainable variable of shape n, regardless of input
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t, scale=1),
            reinterpreted_batch_ndims=1)),
    ])

def posterior_mean_field(kernel_size: int, bias_size: int, dtype: any) -> tf.keras.Model:
    """Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`."""
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))

    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :n],
                       scale=1e-5 + 0.000001*tf.nn.softplus(c + t[..., n:])),
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


#network architecture
#relu stands for rectified linear - modern standard for general application, I think
# model = tf.keras.Sequential([
#     #tf.keras.layers.Conv2D(40,(3,3),activation='relu',input_shape=(5,)),
#     tfp.layers.DenseVariational(320,activation='relu', input_shape=(10,),
#                                 make_posterior_fn=posterior_mean_field,
#                                 make_prior_fn=prior_trainable),
#     tf.keras.layers.Dense(160, activation='relu'),
#     tf.keras.layers.Dense(160, activation='relu'),
#     tf.keras.layers.Dense(80,activation='softmax'),
# ])

# model = tf.keras.Sequential([
#     #tf.keras.layers.Conv2D(40,(3,3),activation='relu',input_shape=(5,)),
#     tfp.layers.DenseVariational(320,activation='relu', input_shape=(10,),
#                                 make_posterior_fn=posterior_mean_field,
#                                 make_prior_fn=prior_trainable),
#     tf.keras.layers.Dense(160, activation='relu'),
#     tf.keras.layers.Dense(160, activation='relu'),
#     tf.keras.layers.Dense(80,activation='softmax'),
#     tf.keras.layers.Concatenate()(),
#     tf.keras.layers.Dense(1),
# ])

input_ = tf.keras.layers.Input(shape=x_train.shape[1:])
hidden1 = tfp.layers.DenseVariational(50, activation='relu', input_shape=(5,),
                                make_posterior_fn=posterior_mean_field,
                                make_prior_fn=prior_trainable)(input_)
hidden2 = tfp.layers.DenseVariational(50, activation='relu', input_shape=(5,),
                                make_posterior_fn=posterior_mean_field,
                                make_prior_fn=prior_trainable)(hidden1)
hidden3 = tfp.layers.DenseVariational(50, activation='relu', input_shape=(5,),
                                make_posterior_fn=posterior_mean_field,
                                make_prior_fn=prior_trainable)(hidden2)
hidden4 = tfp.layers.DenseVariational(50, activation='relu', input_shape=(5,),
                                make_posterior_fn=posterior_mean_field,
                                make_prior_fn=prior_trainable)(hidden3)
concat = tf.keras.layers.Concatenate()([input_, hidden4])
#output = tf.keras.layers.Dense(1)(concat)
distribution_params = tf.keras.layers.Dense(units=2)(concat)
output = tfp.layers.IndependentNormal(1)(distribution_params)
model = tf.keras.Model(inputs=[input_], outputs=[output])




#note sure what these inputs mean. Find out.
#adam utilizes adaptive momentum - variant of stochasticc gradient descent.
#loss is the loss function

#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.compile(optimizer='adam', loss='mean_absolute_error',metrics=[tf.keras.metrics.MeanAbsoluteError()])

model.summary()

history = model.fit(x_train,y_train,batch_size = 1000, epochs=500,shuffle = True,verbose=1,validation_data=(x_test,y_test))
#history = model.fit(x_train,y_train,batch_size=1000,epochs=1000,verbose=1,validation_data=(x_test,y_test))

# probability_model = tf.keras.Sequential([model,
#                                          tf.keras.layers.Softmax()])
#
# predictions = probability_model.predict(x_test)



predictions = model.predict(x_test)
np.argmax(predictions[0])
photoz = []
for i in range(0,len(y_test)):
    photoz.append(predictions[i]*bin_size)

plt.scatter(y_test_original,photoz)
#plt.title('Photo-z determination')
plt.ylabel('spectro-z')
plt.xlabel('photo-z')
plt.show()


# plt.scatter(y_test*bin_size,predictions*bin_size)
# plt.title('Photo-z determination')
# plt.ylabel('spectro-z')
# plt.xlabel('photo-z')
# plt.show()

#
# predictions_test = (model.predict(x_test) > 0.5).astype("int32")
# plt.scatter(y_test,predictions_test)
# plt.title('Photo-z determination')
# plt.ylabel('spectro-z')
# plt.xlabel('photo-z')
# plt.show()
#
#


# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train','Test'],loc='upper left')
# plt.show()

# print("Generate predictions for 3 samples")
# predictions = model.predict(x_test)
#
# plt.scatter(np.array(predictions),np.array(y_test))
# plt.show

# prediction = np.array(prediction)
# z = z[:,0]
#
# ############SHOW STATISTICS############
# err_abs = np.sum(abs(prediction - z)) / z.shape[0]
# deltaz = (prediction - z) / (1 + z)
# bias = np.sum(deltaz) / z.shape[0]
# nmad = 1.48 * np.median(abs(deltaz - np.median(deltaz)))
# print(" N = %d galaxies" %z.size)
# print(" bias = %.4g" %bias)
# print(" sigma_mad = %.4g" %nmad)


#calculate classification error and number of outliers and catastrophic outliers:

# for i = 0, test_set_length-1 do begin
# if output_integer[i] EQ test_target_rounded[i] then ++num_correct
# outliers[i] = (ABS(output[i] - testredshift[i])) / (1.0 + testredshift[i])
# if outliers[i] GT .15 then begin
# outlier_index_default[num_outliers_default + +] = i
# endif
# endfor

num_correct = 0
outliers = []
outlier_index = []
cat_outlier_index = []
#y_test = y_test * bin_size
for i in range(0,len(y_test)):

    if abs(photoz[i] - (y_test[i])*bin_size) < 0.0001:

        num_correct = num_correct + 1

    outliers.append(abs(photoz[i] - y_test_original[i])/(1+y_test_original[i]))

    if outliers[i] > 0.15:
            outlier_index.append(i)


    if outliers[i] > 1:
            cat_outlier_index.append(i)



print("% correct: ", 100.0*num_correct/len(y_test))
print("number of outliers: ", len(outlier_index), " out of ", len(y_test))
print("% of outliers: ", 100.0*len(outlier_index)/len(y_test))
print("number of catastrophic outliers: ", len(cat_outlier_index), " out of ", len(y_test))
print("% of catastrophic outliers: ", len(cat_outlier_index)/len(y_test))

#calculate RMS error:
RMS_error = np.sqrt(np.sum(((abs(photoz - y_test_original)/(1+y_test_original))**2))/len(y_test))
squares = [x*x for x in outliers]
RMS_error_2 = np.sqrt(np.sum(squares)/len(y_test))
print("RMS error: ", RMS_error_2)

#calculating RMSE another way:
RMSE = mean_squared_error(y_test_original,photoz, squared = False)
print("RMS error = ", RMSE)
# ;  if plot EQ 1 then begin
# ;    plot, temporary(testredshift), temporary(output), psym = 1, XRANGE = [0, 4], YRANGE = [0,
#                                                                                             4], title = 'Photo-z vs Spectro-z', xtitle = 'spectroscopic redshift', ytitle = 'photo-z'
# ;    PLOTS, [.18, 1.6 * 2], [0, 1.2 * 2]
# ;    PLOTS, [0, 1.6 * 2], [.15, 2 * 2]
# ;  endif


plt.scatter(y_test_original,photoz, marker='+',color = 'black')
#plt.title('Photo-z determination')
plt.xlabel('spectroscopic redshift')
plt.ylabel('photo z')
plt.plot([.18,1.6*2.4],[0,1.2*2.4], color='black')
plt.plot([0, 1.6 * 2.4], [.15, 2 * 2.4],color = 'black')
plt.show()
