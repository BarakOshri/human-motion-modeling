# For now, the code is used to predict the joints of interest

import os
import numpy as np
import logging
from time import clock
import sys
sys.path.append('/deep/u/kuanfang/human-motion-modeling');

from rnn.rnnl1 import * 
from utils.cornell_utils import *
import utils.my_logging

# Compilation Mode
# theano.config.mode = 'FAST_COMPILE'
theano.config.mode = 'FAST_RUN'
# theano.config.mode = 'DEBUG_MODE'

# Parameters
path_dataset = '../../data/cornell/'
path_models = 'models/'

lr = 1e-3
bs = 1

en_decay = False
epsl_decay = 1e-3

# Load Data
logging.info('Loading data...')
tic = clock()

pos = np.load(os.path.join(path_dataset, 'pos.npy'))
ori = np.load(os.path.join(path_dataset, 'ori.npy'))
data = np.load(os.path.join(path_dataset, 'data.npy'))
index = np.load(os.path.join(path_dataset, 'index.npy'))

toc = clock()
logging.info('Done in %f sec.', toc-tic)

# # Preprocess
# logging.info('Preprocessing data...')
# tic = clock()
# 
# toc = clock()
# logging.info('Done in %f sec.', toc-tic)

# Initialize Model
dimx = data.shape[1]
dimy = data.shape[1]
dimh = 100

logging.info('Initializing model...')
tic = clock()

model = RNNL1(dimx, dimy, dimh)

toc = clock()
logging.info('Done in %f sec.', toc-tic)

# Training 
epoch = 0
prev_cost_train = float('inf')
while 1:
    tic = clock()

    cost_train = 0
    for i in range(index.shape[0]): 
        start = index[i, 0]
        end = index[i, 1]
        # din = np.concatenate([data[start+1:end, :], np.tile(data[end-1, :], (end-1-start, 1))], axis=1)
        din = data[start+1:end, :]
        dout = data[start+1:end, :]
        cost_train += model.train(din,
                                    dout, 
                                    lr)

    cost_train /= (data.shape[0] * data.shape[1])

    toc = clock()
    logging.info('epoch: %d\tcost_train: %f\tlearning_rate: %.2f\ttime: %f', 
                    epoch, cost_train, lr, toc-tic)

    # Learning rate decay
    if en_decay and prev_cost_train - cost_train <= epsl_decay:
        lr /= 2

    prev_cost_train = cost_train

    model.save(os.path.join(path_models, 'model'))

    epoch += 1
