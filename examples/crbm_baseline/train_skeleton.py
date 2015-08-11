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

pos_arr = np.load(os.path.join(path_dataset, 'pos_arr.npy'))
ori_arr = np.load(os.path.join(path_dataset, 'ori_arr.npy'))
data_abs = np.load(os.path.join(path_dataset, 'data_relpos.npy'))
index_abs = np.load(os.path.join(path_dataset, 'index.npy'))

toc = clock()
logging.info('Done in %f sec.', toc-tic)

# Preprocess
# absolute value to incremental value
inits, data, data_index = abs2inc_forall(data_abs, index_abs)

data_mean = np.mean(data, axis=0)
data_std = np.std(data, axis=0)
data = (data - data_mean) / data_std

np.save('data', data)
np.save('inits', inits)
np.save('data_mean', data_mean)
np.save('data_std', data_std)
np.save('data_index', data_index)

# to shared data
batchdata = theano.shared(np.asarray(data, dtype=theano.config.floatX))

# print index
print data_index.shape
seqlen = [data_index[i, 1] - data_index[i, 0]\
            for i in range(data_index.shape[0])]

# direct loss
print 'direct loss: {}'.format(get_loss_direct(data, data_index))

# Training 
crbm, batchdata = train_crbm(learning_rate=1e-2,
                            n_hidden=100,
                            batchdata=batchdata, 
                            seqlen=seqlen, 
                            training_epochs=10000) 
                            # path_model=os.path.join(path_models, 'crbm.npy'))

crbm.save(os.path.join(path_models, 'crbm'))


# # Generating
# data_idx = np.array([100, 200, 400, 600])
# orig_data = numpy.asarray(batchdata.get_value(borrow=True)[data_idx],
#                           dtype=theano.config.floatX)
# 
# hist_idx = np.array([data_idx - n for n in xrange(1, crbm.delay + 1)]).T
# hist_idx = hist_idx.ravel()
# 
# orig_history = numpy.asarray(
#     batchdata.get_value(borrow=True)[hist_idx].reshape(
#     (len(data_idx), crbm.delay * crbm.n_visible)),
#     dtype=theano.config.floatX)
# 
# generated_series = crbm.generate(orig_data, orig_history, n_samples=100,
#                                  n_gibbs=30)
# # append initialization
# generated_series = np.concatenate((orig_history.reshape(len(data_idx),
#                                                         crbm.delay,
#                                                         crbm.n_visible \
#                                                         )[:, ::-1, :],
#                                    generated_series), axis=1)
# 
# bd = batchdata.get_value(borrow=True)
# 
# np.save('generated_series', generated_series)
# np.save('bd', bd)
