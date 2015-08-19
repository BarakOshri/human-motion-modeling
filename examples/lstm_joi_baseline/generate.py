# For now, the code is used to predict the joints of interest

import os
import numpy as np
import logging
from time import clock
import random
import sys
sys.path.append('/deep/u/kuanfang/human-motion-modeling');

from model.rnn import *
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

en_test = True
test_cycle = 50

# Load Data
logging.info('Loading data...')
tic = clock()

pos_arr = np.load(os.path.join(path_dataset, 'pos_arr.npy'))
ori_arr = np.load(os.path.join(path_dataset, 'ori_arr.npy'))
bcpos_arr = np.load(os.path.join(path_dataset, 'bcpos_arr.npy'))
index = np.load(os.path.join(path_dataset, 'index.npy'))

# preprocessed by training code
data_index = np.load('data_index.npy')
index_train = np.load('index_train.npy')
index_test = np.load('index_test.npy')
datain = np.load('datain.npy')
dataout = np.load('dataout.npy')
data_mean = np.load('data_mean.npy')
data_std = np.load('data_std.npy')
dataz = np.load('dataz.npy')
pos_joi_mean = np.load('pos_joi_mean.npy')
pos_joi_std = np.load('pos_joi_std.npy')

toc = clock()
logging.info('Done in %f sec.', toc-tic)

print index_train[-1, 1] - index_train[0, 0]
print dataout[:index_train[-1, 1]].shape

print datain[0:3, :]
print bcpos_arr[0:3, :]

# Initialize Model
logging.info('Initializing model...')
tic = clock()

n_x = datain.shape[1] + dataz.shape[1]
n_y = dataout.shape[1]
n_h = 200
print 'layer size:'
print [n_x, n_h, n_y]

cells = StackedCells(n_x, layers=[n_h], activation=T.tanh, celltype=LSTM)
cells.layers.append(Layer(n_h, n_y, lambda x: x))
model = RNN1LZ(cells, 
                dynamics=lambda x, y: x+y, 
                optimize_method='adadelta')
model.load(os.path.join(path_models, 'model.npy'))

toc = clock()
logging.info('Done in %f sec.', toc-tic)

# # Prediction
# def predict_1(index, i):
#     start = index[i, 0]
#     end = index[i, 1]
#     din = datain[start:end, :]
#     dz = dataz[start:end, :]
#     return model.predict(din, dz)
# 
# pred_train_arr = np.concatenate([predict_1(index_train, i)
#                                 for i in range(index_train.shape[0])],
#                                 axis=0)
# gt_train_arr = np.concatenate([dataout[index_train[i, 0]:index_train[i, 1]]
#                                 for i in range(index_train.shape[0])],
#                                 axis=0)
# print 'prediction loss on trainset: %f'\
#         % np.sqrt(np.mean((pred_train_arr - gt_train_arr) ** 2))
# 
# pred_test_arr = np.concatenate([predict_1(index_test, i)
#                                 for i in range(index_test.shape[0])],
#                                 axis=0)
# gt_test_arr = np.concatenate([dataout[index_test[i, 0]:index_test[i, 1]]
#                                 for i in range(index_test.shape[0])],
#                                 axis=0)
# print 'prediction loss on testset: %f'\
#         % np.sqrt(np.mean((pred_test_arr - gt_test_arr) ** 2))

seq_idx = [0, 1, 2, 3]
seq_idx = [0]
len_seed = 10
list_gen_serie = []
for i in seq_idx:
    start = index_train[i, 0]
    end = index_train[i, 1]
    print 'frame range: {}'.format((start, end))
    din = datain[start:start+len_seed, :]
    dz = dataz[start:end, :]

    gen_serie = model.generate(din, dz) 
    gen_serie = np.concatenate([din, gen_serie], axis=0)
    list_gen_serie.append(gen_serie)
    print 'size of generated serie: {}'.format(gen_serie.shape)

    np.save(os.path.join(path_outputs, 'gen_serie_{}'.format(i)), 
            gen_serie)
    logging.info('saved gen_serie_%d', i)
    


