from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
VQ PM. uses
pip install tensorflow-gpu==2.0.0-beta1
pip install git+https://github.com/deepmind/sonnet@v2
see vqvae2_singleMin_PM colab

See original code: https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb
"""
import os
import subprocess
import tempfile

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import tarfile
import pandas as pd

from six.moves import cPickle
from six.moves import urllib
from six.moves import xrange

from vq_model import VAEModel
from tensorflow.keras import losses
from common import whole_paths
#============================================================================================================
PM_FILE_TRAIN = '012018-062019_raw_primemap_train.csv'
PM_FILE_VALID = '012018-062019_raw_primemap_valid.csv'
CHKPT_DIR = 'vq_mod_8nembed'
SAVE_INTERVAL = 250

RECON_COEF = 1.0
# Set hyper-parameters.
VALID_FRAC = 10 # @param (10 means 1/10 is test)
BATCH_SIZE = 32
# 100k steps should take < 30 minutes on a modern (>= 2017) GPU.
NUM_TRAINING_UPDATES = 10000
KWARGS ={}

KWARGS['in_channel']=1
KWARGS['main_channel']=32
KWARGS['num_res_blocks']=2
KWARGS['residual_hiddens']=8
# This value is not that important, usually 64 works.
# This will not change the capacity in the information-bottleneck.
KWARGS['embed_dim']=16
# The higher this value, the higher the capacity in the information bottleneck.
KWARGS['n_embed']=8
KWARGS['decay']=0.99
KWARGS['commitment_cost']=0.25
# commitment_cost should be set appropriately. It's often useful to try a couple
# of values. It mostly depends on the scale of the reconstruction cost
# (log p(x|z)). So if the reconstruction cost is 100x higher, the
# commitment_cost should also be multiplied with the same amount.

# Use EMA updates for the codebook (instead of the Adam optimizer).
# This typically converges faster, and makes the model less dependent on choice
# of the optimizer. In the VQ-VAE paper EMA updates were not used (but was
# developed afterwards). See Appendix of the paper for more details.
# This is only used for EMA updates.

LEARNING_RATE = 3e-4

#=============================================================================================================
PM_FILE_TRAIN,PM_FILE_VALID,CHKPT_DIR = whole_paths([PM_FILE_TRAIN,PM_FILE_VALID,CHKPT_DIR])


def log_and_normalise(data_mat):#,normalizer):
  """Log data range [-18.42, 0.0] (softmax at end)
  transform to [0.0 to 1.0]
  """
  data_mat = tf.cast(data_mat,dtype=tf.float32)
  far_val = tf.constant(18.43,dtype=tf.float32)
  data_mat = tf.math.log(data_mat + 1e-8)
  data_mat = data_mat/ far_val +1
  data_mat= tf.expand_dims(data_mat,-1)
  return data_mat

def np_var(data_mat):
    return  np.var(np.log((data_mat+1e-8))/18.43)

train_data = np.array(pd.read_csv(PM_FILE_TRAIN, sep=',',header=None).values,dtype=np.float32)
valid_data = np.array(pd.read_csv(PM_FILE_VALID, sep=',',header=None).values,dtype=np.float32)

#train_data_var = np_var(train_data)
# Data Loading.
train_dataset_iterator = iter(
    tf.data.Dataset.from_tensor_slices(train_data)
    .map(log_and_normalise)
    .shuffle(100000)
    .repeat(-1)  # repeat indefinitely
    .batch(BATCH_SIZE))
valid_dataset_iterator = iter(
    tf.data.Dataset.from_tensor_slices(valid_data)
    .map(log_and_normalise)
    .repeat(1)  # 1 epoch
    .batch(BATCH_SIZE))
def get_images(subset='train'):
  if subset == 'train':
    return next(train_dataset_iterator)
  elif subset == 'valid':
    return next(valid_dataset_iterator)

#=========================================================================

opt2 = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)



vae_all = VAEModel(**KWARGS)
# Initialize Shape
inx = get_images('train')
vae_all(inx)

checkpoint = tf.train.Checkpoint(module = vae_all)
# Process inputs with conv stack, finishing with 1x1 to get to correct size.
def get_recon_error(x,x_recon):
    x = tf.math.softmax(tf.squeeze(x,-1),axis=-1)
    x_recon = tf.math.softmax(tf.squeeze(x_recon,-1),axis=-1)
    loss = losses.kullback_leibler_divergence(x,x_recon)
    return loss
@tf.function
def train_step(x,recon_coef =1.0):
  with tf.GradientTape() as tape:


    x_recon,diff,quants=vae_all(x,is_training=True)
    #recon_error = tf.reduce_mean((x_recon - x)**2) /train_data_var
    recon_error = tf.reduce_mean(get_recon_error(x,x_recon)) * recon_coef
    
    loss = diff+recon_error
  params= vae_all.trainable_variables
  grads = tape.gradient(loss, params)
  opt2.apply_gradients(zip(grads, params))
  return recon_error,quants[0]["perplexity"],quants[1]["perplexity"]


# Train.
train_res_recon_error = []
train_res_perplexity_t = []
train_res_perplexity_b = []
for i in xrange(NUM_TRAINING_UPDATES):
  inx = get_images('train')
  results = train_step(inx,RECON_COEF)
  train_res_recon_error.append(results[0])
  train_res_perplexity_b.append(results[2])
  train_res_perplexity_t.append(results[1])

  if (i+1) % SAVE_INTERVAL == 0:
    print('%d iterations' % (i+1))
    print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
    print('perplexity: %.3f' % np.mean(train_res_perplexity_t[-100:]))
    print('perplexity: %.3f' % np.mean(train_res_perplexity_b[-100:]))
    checkpoint.save(os.path.join(CHKPT_DIR,'ver'))
checkpoint.save(os.path.join(CHKPT_DIR,'ver'))
f = plt.figure(figsize=(16,8))
ax = f.add_subplot(1,3,1)
ax.plot(train_res_recon_error)
ax.set_yscale('log')
ax.set_title('NMSE.')

ax = f.add_subplot(1,3,2)
ax.plot(train_res_perplexity_t)
ax.set_title('Average codebook usage (perplexity).')

ax = f.add_subplot(1,3,3)
ax.plot(train_res_perplexity_b)
ax.set_title('Average codebook usage (perplexity).')
plt.show()
