# For now, the code is used to predict the joints of interest

import os
import numpy as np
import logging
from time import clock
import random
import sys
sys.path.append('/deep/u/kuanfang/human-motion-modeling');

from model.crbm import *
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


# Load Data
data = np.load('data.npy')
data_mean = np.load('data_mean.npy')
data_std = np.load('data_std.npy')
data_index = np.load('data_index.npy')

# Preprocess
batchdata = theano.shared(np.asarray(data, dtype=theano.config.floatX))

x = T.matrix('x')  # the data
x_history = T.matrix('x_history')
learning_rate=1e-3 
training_epochs=300
batch_size=100
n_dim = batchdata.get_value(borrow=True).shape[1]
n_hidden=100 
delay=6

seq_idx = [0, 1, 2, 3]
# for i in seq_idx:
#     print 'seq {}, length {}'\
#         .format(i, data_index[i, 1] - data_index[i, 0] - delay)

data_idx = np.array([data_index[i, 0]+delay for i in seq_idx])
orig_data = numpy.asarray(batchdata.get_value(borrow=True)[data_idx],
                          dtype=theano.config.floatX)

hist_idx = np.array([data_idx - n for n in xrange(1, delay + 1)]).T
hist_idx = hist_idx.ravel()

orig_history = numpy.asarray(
                batchdata.get_value(borrow=True)[hist_idx].reshape(
                (len(data_idx), delay * n_dim)),
                dtype=theano.config.floatX)

# Generation
crbm = CRBM(input=x, input_history=x_history, n_visible=n_dim, \
            n_hidden=n_hidden, delay=delay)
crbm.load(os.path.join(path_models, 'crbm.npy'))

generated_series = crbm.generate(orig_data, orig_history, n_samples=100, 
                                    n_gibbs=30)
generated_series = np.concatenate((orig_history.reshape(len(data_idx),
                                                        crbm.delay,
                                                        crbm.n_visible \
                                                        )[:, ::-1, :],
                                   generated_series), axis=1)

for i in range(generated_series.shape[0]):
    np.save(os.path.join(path_outputs, 'gen_series_{}'.format(i)), 
            generated_series[i])

# print orig_history.reshape(len(data_idx), crbm.delay, crbm.n_visible)[:, ::-1, :]
# print data[data_index[seq_idx, 0], :]
