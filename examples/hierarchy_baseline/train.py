# For now, the code is used to predict the joints of interest

import os
import numpy as np
import logging
from time import clock
import random
import sys
sys.path.append('/deep/u/kuanfang/human-motion-modeling');

from rnn.rnn import * 
from utils.cornell_utils import *
import utils.my_logging

# Compilation Mode
# theano.config.mode = 'FAST_COMPILE'
theano.config.mode = 'FAST_RUN'
# theano.config.mode = 'DEBUG_MODE'

# Parameters
path_dataset = '../../data/cornell/'
path_models = 'models/'

lr = 1e-0
bs = 1

en_decay = False
epsl_decay = 1e-3

en_test = True

# Load Data
logging.info('Loading data...')
tic = clock()

pos = np.load(os.path.join(path_dataset, 'pos.npy'))
ori = np.load(os.path.join(path_dataset, 'ori.npy'))
data = np.load(os.path.join(path_dataset, 'data.npy'))
mean = np.load(os.path.join(path_dataset, 'mean.npy'))
std = np.load(os.path.join(path_dataset, 'std.npy'))
pos_joi = np.load(os.path.join(path_dataset, 'pos_joi.npy'))
index = np.load(os.path.join(path_dataset, 'index.npy'))

toc = clock()
logging.info('Done in %f sec.', toc-tic)

# Preprocess
num_seq_train = 50
index_train = index[0:num_seq_train, :]
index_test = index[num_seq_train:, :]
len_train = sum([index[i, 1] - index[i, 0] 
                for i in range(num_seq_train)])
len_test = sum([index[i, 1] - index[i, 0] 
                for i in range(num_seq_train, index.shape[0])])

# Variables for Evaluation
# ground truth
d_train = np.concatenate([data[index_train[i, 0]+1:index_train[i, 1], :] 
                           for i in range(index_train.shape[0])], axis=0)
d_test = np.concatenate([data[index_test[i, 0]+1:index_test[i, 1], :] 
                            for i in range(index_test.shape[0])], axis=0)

# direct loss
def get_loss_direct(data, index):
    loss_direct = 0
    for i in range(index.shape[0]):
        start = index[i, 0]
        end = index[i, 1]
        din = data[start:end-1, :]
        dout = data[start+1:end, :]
        loss_direct += np.sum((dout - din)**2) / data.shape[1]
    loss_direct = np.sqrt(loss_direct / index.shape[0])
    return loss_direct

print 'direct loss:'
print get_loss_direct(data, index_train)
print get_loss_direct(data, index_test) 


# Initialize Model
dim_x = data.shape[1]
dim_y = data.shape[1]
dim_h = 5

logging.info('Initializing model...')
tic = clock()

model = RNNL1(dim_x, dim_h, dim_y)

toc = clock()
logging.info('Done in %f sec.', toc-tic)

# Training 
def predict_seq(model, data, index, i):
    """
    Predict a sequence.
    """
    start = index[i, 0]
    end = index[i, 1]
    din = data[start:end-1, :]
    return model.predict(din)

epoch = 0
prev_loss_train = float('inf')

# while 1:
while epoch <= 100:
    tic = clock()

    # Shuffle
    _index_train = index_train.copy()
    np.random.shuffle(_index_train) 

    loss_train = 0
    for i in range(_index_train.shape[0]): 
        start = _index_train[i, 0]
        end = _index_train[i, 1]
        din = data[start:end-1, :]
        dout = data[start+1:end, :]
        loss_train += model.train(din, dout, lr)

    loss_train = np.sqrt(loss_train/index_train.shape[0])

    toc = clock()
    logging.info('epoch: %d\tloss_train: %f\tlearning_rate: %.2f\ttime: %f', 
                    epoch, loss_train, lr, toc-tic)

    # Learning rate decay
    if en_decay and prev_loss_train - loss_train <= epsl_decay:
        lr /= 2

    # Test
    if (en_test):
        preds_train = np.concatenate([predict_seq(model, data, index_train, i) 
                                for i in range(index_train.shape[0])], axis=0)
        preds_test = np.concatenate([predict_seq(model, data, index_test, i) 
                                for i in range(index_test.shape[0])], axis=0)
        print np.sqrt(np.mean((preds_train - d_train)**2))
        print np.sqrt(np.mean((preds_test - d_test)**2))

    # 
    prev_loss_train = loss_train

    model.save(os.path.join(path_models, 'model'))

    epoch += 1

# Evaluate the generation
n_seed = 10
n_gen = 10
y_gen= model.generate(data[0:n_seed, :], n_gen)
print 'seed:'
print data[0:n_seed, :]
print 'ground truch:'
print data[n_seed:n_seed+n_gen, :]
print 'generated sequence:'
print y_gen
print 'loss_direct: {}'.format(
    np.sqrt(np.mean((data[n_seed:n_seed+n_gen, :] - \
            np.tile(data[n_seed-1, :], (n_gen, 1)))**2)))
print 'loss_gen: {}'.format(
    np.sqrt(np.mean((data[n_seed:n_seed+n_gen, :] - y_gen)**2)))

for t in range(n_gen):
    print 'step: {}'.format(t)
    print 'loss_direct: {}'\
            .format(np.sqrt(np.mean((data[n_seed+t, :] - data[n_seed-1, :])**2)))
    print 'loss_gen: {}'\
            .format(np.sqrt(np.mean((data[n_seed+t, :] - y_gen[t, :])**2)))
