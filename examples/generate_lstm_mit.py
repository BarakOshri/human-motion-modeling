import os
import numpy as np
import logging
from time import clock
import random
import sys
sys.path.append('../');

from model.rnn import *
from util.mit_utils import *
import util.my_logging
from util.mocap_utils import *
import pdb
from util.motion import load_data

# Compilation Mode
theano.config.mode = 'FAST_COMPILE'
# theano.config.mode = 'FAST_RUN'
# theano.config.mode = 'DEBUG_MODE'
# theano.config.exception_verbosity = 'high'

output_subdirectory = 'predframehidden256lr0.8tanh'

# Parameters
path_dataset = '../data/mit/'
path_models = '../model/'
path_outputs = '../outputs/' + output_subdirectory

lr = 1e-1 # leaning rate
bs = 1 # batch size

en_decay = False # enable learning rate decay
epsl_decay = 1e-3 # epsilon value of learning rate decy

en_test = True
test_cycle = 50

# Load Data
logging.info('Loading data...')
tic = clock()

x, seqlen, data_mean, data_std, offsets = load_data('../data/mit/motion.mat', shared=False)
datain = x

# preprocessed by training code
#datain = np.load(os.path.join(path_outputs,'datain.npy'))
dataout = np.load(os.path.join(path_outputs,'dataout.npy'))
data_mean = np.load(os.path.join(path_outputs,'data_mean.npy'))
data_std = np.load(os.path.join(path_outputs,'data_std.npy'))
offsets = np.load(os.path.join(path_outputs,'offsets.npy'))

toc = clock()
logging.info('Done in %f sec.', toc-tic)

# Initialize Model
logging.info('Initializing model...')
tic = clock()

n_x = datain.shape[1]
n_y = dataout.shape[1]
n_h = 200
print 'layer size:'
print [n_x, n_h, n_y]

cells = StackedCells(n_x, layers=[n_h], activation=T.tanh, celltype=LSTM)
cells.layers.append(Layer(n_h, n_y, lambda x: x))
model = RNN1L(cells, 
                dynamics=lambda x, y: x+y, 
                optimize_method='adadelta')
model.load(os.path.join(path_outputs, 'model.npy'))

toc = clock()
logging.info('Done in %f sec.', toc-tic)

n_generate = 55
seq_idx = [0, 1, 2]
seq_idx_start = [0, 50, 100]
len_seed = 10
list_gen_serie = []
for i in seq_idx:
    start = seq_idx_start[i]
    print 'frame range: {}'.format((start, start+len_seed))
    din = datain[start:start+len_seed, :]

    gen_serie = model.generate(din, n_generate) 
    gen_serie = np.concatenate([din, gen_serie], axis=0)
    list_gen_serie.append(gen_serie)
    print 'size of generated serie: {}'.format(gen_serie.shape)

    pdb.set_trace()
    gen_serie = postprocess(gen_serie, data_std.reshape((1, -1)), data_mean.reshape((1, -1)), offsets)
    print gen_serie.shape
    np.save(os.path.join(path_outputs, 'gen_serie_{}'.format(i)), 
            gen_serie)
    logging.info('saved gen_serie_%d', i)
    


