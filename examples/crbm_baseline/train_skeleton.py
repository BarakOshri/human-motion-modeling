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
# absolute value to incremental value
data, init_frame = abs2incr(data, index)

# print index.shape
# data_incr, init_frame = abs2incr(data, index)
# data_recon = incr2abs(data_incr, init_frame, index)
# print np.mean((data - data_recon)**2)


# to shared data
batchdata = theano.shared(np.asarray(data, dtype=theano.config.floatX))

# print index
# seqlen = [index[i, 1] - index[i, 0] for i in range(index.shape[0])]
seqlen = [index[i, 1] - index[i, 0] - 1 for i in range(index.shape[0])]
print sum(seqlen)
    

# direct loss
print 'direct loss: {}'.format(get_loss_direct(data, index))

# Training 
crbm, batchdata = train_crbm(batchdata=batchdata, 
                            seqlen=seqlen, 
                            training_epochs=500) 
                            # path_model=os.path.join(path_models, 'crbm.npy'))


