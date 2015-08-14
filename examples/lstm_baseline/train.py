# For now, the code is used to predict the joints of interest

import os
import numpy as np
import logging
from time import clock
import random
import sys
sys.path.append('/deep/u/kuanfang/human-motion-modeling');

import theano
from model.lstm import *
from util.cornell_utils import *
import util.my_logging
from util.mocap_utils import *

# Compilation Mode
# theano.config.mode = 'FAST_COMPILE'
theano.config.mode = 'FAST_RUN'
# theano.config.mode = 'DEBUG_MODE'
# theano.config.exception_verbosity = 'high'

# Parameters
path_dataset = '../../data/cornell/'
path_models = 'models/'
path_outputs = 'outputs/'

lr = 1e-1 # leaning rate
bs = 1 # batch size

en_decay = False # enable learning rate decay
epsl_decay = 1e-3 # epsilon value of learning rate decy

# Load Data
logging.info('Loading data...')
tic = clock()

pos_arr = np.load(os.path.join(path_dataset, 'pos_arr.npy'))
ori_arr = np.load(os.path.join(path_dataset, 'ori_arr.npy'))
bcpos_arr = np.load(os.path.join(path_dataset, 'bcpos_arr.npy'))
index = np.load(os.path.join(path_dataset, 'index.npy'))

toc = clock()
logging.info('Done in %f sec.', toc-tic)

# Preprocess
data_mean = np.mean(bcpos_arr, axis=0)
data_std = np.std(bcpos_arr, axis=0)
bcpos_arr_ = (bcpos_arr - data_mean) / data_std

datain = np.concatenate(
            [bcpos_arr_[index[i, 0]:index[i, 1]-1, :]\
            for i in range(index.shape[0])],
            axis=0).astype('float32')
dataout = np.concatenate(
            [bcpos_arr_[index[i, 0]+1:index[i, 1], :]\
            for i in range(index.shape[0])],
            axis=0).astype('float32')
data_index = np.concatenate(
                [[[index[i, 0]-i, index[i, 1]-i-1]]\
                for i in range(index.shape[0])],
                axis=0)
# inits, dataout, data_index = abs2inc_forall(bcpos_arr_, index)

index_train = data_index.copy()
index_test = data_index.copy()

# np.save('inits', inits)
np.save('data_index', data_index)

np.save('datain', datain)
np.save('dataout', dataout)
np.save('data_mean', data_mean)
np.save('data_std', data_std)

# Initialize Model
logging.info('Initializing model...')
tic = clock()

n_x = datain.shape[1]
n_y = datain.shape[1]
n_h = 100
print [n_x, n_y, n_h]
# dropout = 0.0

model = LSTML1(n_x, n_h, n_y, dynamics=lambda x, y: x+y)

toc = clock()
logging.info('Done in %f sec.', toc-tic)

# Training 
list_loss = []
for i in range(index_train.shape[0]): 
    start = index_train[i, 0]
    end = index_train[i, 1]
    din = datain[start:end, :]
    dout = dataout[start:end, :]
    loss = np.mean((din - dout) ** 2)
    list_loss.append(loss)

loss_train = np.sqrt(np.mean(list_loss))
print 'direct loss: %f' % loss_train

epoch = 0
prev_loss_train = float('inf')

# while 1:
while epoch <= 5000:
    tic = clock()

    # Shuffle
    index_train_ = index_train.copy()
    # np.random.shuffle(index_train_) 

    list_loss = []
    for i in range(index_train_.shape[0]): 
        start = index_train_[i, 0]
        end = index_train_[i, 1]
        din = datain[start:end, :]
        dout = dataout[start:end, :]
        list_loss.append(model.train(din, dout))

    loss_train = np.sqrt(np.mean(list_loss))

    toc = clock()
    logging.info('epoch: %d\tloss_train: %f\tlearning_rate: %.2f\ttime: %f', 
                    epoch, loss_train, lr, toc-tic)

    # Learning rate decay
    if en_decay and prev_loss_train - loss_train <= epsl_decay:
        lr /= 2

    # 
    prev_loss_train = loss_train

    model.save(os.path.join(path_models, 'model'))

    epoch += 1

