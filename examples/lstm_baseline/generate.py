# For now, the code is used to predict the joints of interest

import os
import numpy as np
import logging
from time import clock
import random
import sys
sys.path.append('/deep/u/kuanfang/human-motion-modeling');

from model.lstm import *
from util.cornell_utils import *
import util.my_logging
from util.mocap_utils import *

# Compilation Mode
theano.config.mode = 'FAST_COMPILE'
# theano.config.mode = 'FAST_RUN'
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

index_train = data_index.copy()
index_test = data_index.copy()

# Initialize Model
logging.info('Initializing model...')
tic = clock()

n_x = datain.shape[1]
n_y = datain.shape[1]
n_h = 100
print [n_x, n_y, n_h]

model = LSTML1(n_x, n_h, n_y, dynamics=lambda x, y: x+y)
model.load(os.path.join(path_models, 'model.npy'))

toc = clock()
logging.info('Done in %f sec.', toc-tic)

# Prediction
def predict_1(i):
    start = index_train[i, 0]
    end = index_train[i, 1]
    din = datain[start:end, :]
    return model.predict(din)

pred_arr = np.concatenate([predict_1(i) for i in range(index_train.shape[0])],
                            axis=0)
print 'prediction loss: %f' % np.mean((pred_arr - dataout) ** 2)
# print pred_arr[10:20]
# print dataout[10:20]

data_idx = [0, 1, 2, 3]
len_seed = 10
n_gen = 100
list_gen_serie = []
for i in data_idx:
    start = index_train[i, 0]
    end = index_train[i, 1]
    din = datain[start:start+len_seed, :]

    gen_serie = model.generate(din, n_gen) 
    print gen_serie.shape
    gen_serie = np.concatenate([din, gen_serie], axis=0)
    list_gen_serie.append(gen_serie)

    np.save(os.path.join(path_outputs, 'gen_serie_{}'.format(i)), 
            gen_serie)
    logging.info('saved gen_serie_%d', i)
    


