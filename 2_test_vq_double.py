from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
VQ PM. review model
"""
import os

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import pandas as pd

#from vq_model import VAEModel
from vq_model import VAEModel
from scipy.special import softmax
#============================================================================================================
num_views = 4# @params
view_trains = False

PM_FILE_TRAIN = '012018-062019_raw_primemap_train.csv'
PM_FILE_VALID = '012018-062019_raw_primemap_valid.csv'
CHKPT_DIR =  'vq_mod_8nembed'
VER_DIR = 'ver-41'
# Set hyper-parameters.
VALID_FRAC = 10 # @param (10 means 1/10 is test)
BATCH_SIZE = 64
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
#==============================================================================================================
dir_path = os.path.dirname(os.path.realpath(__file__))
PM_FILE_TRAIN = os.path.join(dir_path,PM_FILE_TRAIN)
PM_FILE_VALID = os.path.join(dir_path,PM_FILE_VALID)
CHKPT_DIR = os.path.join(dir_path,CHKPT_DIR)


train_data = np.array(pd.read_csv(PM_FILE_TRAIN, sep=',',header=None).values,dtype=np.float32)
valid_data = np.array(pd.read_csv(PM_FILE_VALID, sep=',',header=None).values,dtype=np.float32)
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


vae_all = VAEModel(**KWARGS)
inx = get_images('train')
vae_all(inx)

checkpoint = tf.train.Checkpoint(module = vae_all)
checkpoint.restore(os.path.join(CHKPT_DIR,VER_DIR))


#===========================================VIEW RECONSTRUCTIONS===================================
def do_x_recon(x):
    #x_recon,_,_,_=vae_all(x,is_training=False)
    x_recon,_,_=vae_all(x,is_training=False)
    return x_recon
def revert_transform(image_batch):
  image_batch = np.array(image_batch,dtype=np.float32)
  images = np.squeeze(image_batch,-1)
  images-=1
  images*=18.43
  images = softmax(images,axis=-1)
  return images

xvals = np.linspace(-8,7,16)

train_originals = get_images('train')
train_reconstructions = do_x_recon(train_originals)


valid_originals = get_images('valid')
valid_reconstructions = do_x_recon(valid_originals)

viewinds = np.arange(0,BATCH_SIZE)

for _ in range(5):
    #plt.cla()
    
    np.random.shuffle(viewinds)
    f = plt.figure(figsize=(8,8))
    for i in range(num_views):
        
        if view_trains:
          tempY = revert_transform(train_originals[viewinds[i]])
          tempY2 =revert_transform(train_reconstructions[viewinds[i]])
        else:
          tempY = revert_transform(valid_originals[viewinds[i]])
          tempY2 =revert_transform(valid_reconstructions[viewinds[i]])
        ax = f.add_subplot(num_views,2,i*2+1)
        ax.bar(xvals,np.round(tempY*60))
        #ax.set_title('training data originals')

        ax = f.add_subplot(num_views,2,i*2+2)
        ax.bar(xvals,np.round(tempY2*60))

    plt.show()
    



#c=vae_all.transform(train_originals)
#print("New representation", c.shape, c[0])
_,_,quants=vae_all(train_originals,is_training=False)
quant_t,quant_b = quants
qt,qb = quant_t["encoding_indices"].numpy(),quant_b["encoding_indices"].numpy()
print("Int Representations")
for t,b in zip(qt,qb):
    print(t,b)
#print("Int Representations",quant_b["encoding_indices"].shape,quant_b["encoding_indices"])
#print("Int Representations",quant_t["encoding_indices"].shape,quant_t["encoding_indices"])