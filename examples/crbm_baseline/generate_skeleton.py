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
data = np.load(os.path.join(path_dataset, 'data_relpos.npy'))
mean = np.load(os.path.join(path_dataset, 'mean_relpos.npy'))
std = np.load(os.path.join(path_dataset, 'std_relpos.npy'))
index = np.load(os.path.join(path_dataset, 'index.npy'))

toc = clock()
logging.info('Done in %f sec.', toc-tic)

# Preprocess
data, init_frame = abs2incr(data, index)

batchdata = theano.shared(np.asarray(data, dtype=theano.config.floatX))
print theano.config.floatX

# Generation
path_models = 'models/'
x = T.matrix('x')  # the data
x_history = T.matrix('x_history')
learning_rate=1e-3 
training_epochs=300
batch_size=100
n_dim = batchdata.get_value(borrow=True).shape[1]
n_hidden=100 
delay=6
crbm = CRBM(input=x, input_history=x_history, n_visible=n_dim, \
            n_hidden=n_hidden, delay=delay)

crbm.load(os.path.join(path_models, 'crbm.npy'))

print index[0, 0]
data_idx = np.array([index[0, 0]+delay, index[1, 0]+delay-1, index[2, 0]+delay-2, index[3, 0]+delay-3])
orig_data = numpy.asarray(batchdata.get_value(borrow=True)[data_idx],
                          dtype=theano.config.floatX)

hist_idx = np.array([data_idx - n for n in xrange(1, crbm.delay + 1)]).T
hist_idx = hist_idx.ravel()

orig_history = numpy.asarray(
                batchdata.get_value(borrow=True)[hist_idx].reshape(
                (len(data_idx), crbm.delay * crbm.n_visible)),
                dtype=theano.config.floatX)

generated_series = crbm.generate(orig_data, orig_history, n_samples=100, 
                                    n_gibbs=30)
generated_series = np.concatenate((orig_history.reshape(len(data_idx),
                                                        crbm.delay,
                                                        crbm.n_visible \
                                                        )[:, ::-1, :],
                                   generated_series), axis=1)

for i in range(generated_series.shape[0]):
    serie = generated_series[i, :, :]
    serie = incr2abs(serie, init_frame[[i]], np.array([[0, 101]]))
    print serie.shape
    pos_gen = postprocess_relpos(joint_idx, connection, serie, mean, std)
    np.save(os.path.join(path_outputs, 'pos_gen_{}'.format(i)), pos_gen)

print generated_series[0, :, :]
print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
print incr2abs(generated_series[0, :, :], init_frame[[0]], np.array([[0, 101]]))


