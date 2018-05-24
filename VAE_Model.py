#***********************************************************************
#Author: Ayse Dincer
#Date: 23 May 2018
#Keras implementation of VAE for DeepProfile. Paper is available: https://www.biorxiv.org/content/early/2018/03/08/278739
#Code is modified from https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
#***********************************************************************

import os
import numpy as np
import pandas as pd
import math 
import csv
import sys

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.layers import Input, Dense, Lambda, Layer, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras import metrics, optimizers
from keras.callbacks import Callback
import keras

#Reparameterization trick 
def sampling(args):
    
    z_mean, z_log_var = args

    epsilon = K.random_normal(shape=K.shape(z_mean), mean=0., stddev=1.0)
    
    z = z_mean + K.exp(z_log_var / 2) * epsilon
    return z

#Vae loss defined
def vae_loss(x_input, x_decoded):
    reconstruction_loss = original_dim * metrics.mse(x_input, x_decoded)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    print K.get_value(beta)
    return K.mean(reconstruction_loss + (K.get_value(beta) * kl_loss))

#Reconstruction loss defined
def reconstruction_loss(x_input, x_decoded):
    return metrics.mse(x_input, x_decoded)

#KL loss defined
def kl_loss(x_input, x_decoded):
    return - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

class WarmUpCallback(Callback):
    def __init__(self, beta, kappa):
        self.beta = beta
        self.kappa = kappa
    
    # Behavior on each epoch
    def on_epoch_end(self, epoch, logs={}):
        if K.get_value(self.beta) <= 1:
            K.set_value(self.beta, K.get_value(self.beta) + self.kappa)

#Read input file
input_filename = sys.argv[1]
output_filename = sys.argv[2]
input_df = pd.read_table(input_filename, index_col=0)
print "INPUT FILE..."
print input_df.shape 
print input_df.head(5)

# Set hyperparameters
original_dim = input_df.shape[1]
intermediate1_dim = int(sys.argv[3])
intermediate2_dim = int(sys.argv[4])
latent_dim = int(sys.argv[5])

batch_size = 50
learning_rate = 0.0005
beta = K.variable(1)
kappa = 0

test_data_size = int(sys.argv[6])
epochs = int(sys.argv[7])
fold_count = int(sys.argv[8])

#Separate data to training and test sets
input_df_training = input_df.iloc[:-1 * test_data_size, :]
input_df_test = input_df.iloc[-1 * test_data_size:, :]

print "INPUT DF"
print input_df_training.shape
print input_df_training.index
print "TEST DF"
print input_df_test.shape
print input_df_test.index

#Define encoder
x = Input(shape=(original_dim, ))

net = Dense(intermediate1_dim)(x)
net2 = BatchNormalization()(net)
net3 = Activation('relu')(net2)

net4 = Dense(intermediate2_dim)(net3)
net5 = BatchNormalization()(net4)
net6 = Activation('relu')(net5)

z_mean = Dense(latent_dim)(net6)
z_log_var = Dense(latent_dim)(net6)

# Sample from mean and var
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

#Define decoder
decoder_h = Dense(intermediate2_dim, activation='relu')
decoder_h2 = Dense(intermediate1_dim, activation='relu')
decoder_mean = Dense(original_dim)

h_decoded = decoder_h(z)
h_decoded2 = decoder_h2(h_decoded)
x_decoded_mean = decoder_mean(h_decoded2)

#VAE model
vae = Model(x, x_decoded_mean)

adam = optimizers.Adam(lr=learning_rate)
vae.compile(optimizer=adam, loss = vae_loss, metrics = [reconstruction_loss, kl_loss])
vae.summary()

#Train from only training data
history  = vae.fit(np.array(input_df_training), np.array(input_df_training),
               shuffle=True,
               epochs=epochs,
               batch_size=batch_size,
               verbose = 2,
               validation_data=(np.array(input_df_test), np.array(input_df_test)),
               callbacks=[WarmUpCallback(beta, kappa)])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('VAE Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
fig = plt.gcf()
fig.set_size_inches(14.5, 8.5)
plt.show()

plt.plot(history.history['reconstruction_loss'])
plt.plot(history.history['val_reconstruction_loss'])
plt.title('VAE Model Reconstruction Error')
plt.ylabel('reconstruction error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
fig = plt.gcf()
fig.set_size_inches(14.5, 8.5)
plt.show()


# DEFINE ENCODER
encoder = Model(x, z_mean)

#SAVE THE ENCODER
from keras.models import model_from_json

model_json = encoder.to_json()
with open("encoder" + str(fold_count) + ".json", "w") as json_file:
    json_file.write(model_json)

encoder.save_weights("encoder" + str(fold_count) + ".h5")
print("Saved model to disk")


#DEFINE DECODER
decoder_input = Input(shape=(latent_dim, )) 
_h_decoded = decoder_h(decoder_input)
_h_decoded2 = decoder_h2(_h_decoded)
_x_decoded_mean = decoder_mean(_h_decoded2)
decoder = Model(decoder_input, _x_decoded_mean)

# Encode test data into the latent representation - and save output
test_encoded = encoder.predict(input_df_test, batch_size = batch_size)
test_encoded_df = pd.DataFrame(test_encoded, index = input_df_test.index)
test_encoded_df.to_csv(output_filename + str(fold_count) + ".tsv", sep='\t', quoting = csv.QUOTE_NONE)

# How well does the model reconstruct the input data
test_reconstructed = decoder.predict(np.array(test_encoded_df))
test_reconstructed_df = pd.DataFrame(test_reconstructed, index = input_df_test.index, columns = input_df_test.columns)

recons_error = mean_squared_error(np.array(input_df_test), np.array(test_reconstructed_df))

print("TEST RECONSTRUCTION ERROR: " + str(recons_error))