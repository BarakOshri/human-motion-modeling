# For now, the code is used to predict the joints of interest

import os
import numpy as np
import theano
import theano.tensor as T
import logging
from time import clock
import random
import sys
sys.path.append('/deep/u/kuanfang/human-motion-modeling');

from model.rnn import *
from util.cornell_utils import *
import util.my_logging
from util.mocap_utils import *

# Compilation Configuration
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

en_test = True
test_cycle = 10

# Load Data
logging.info('Loading data...')
tic = clock()

pos_arr = np.load(os.path.join(path_dataset, 'pos_arr.npy'))
ori_arr = np.load(os.path.join(path_dataset, 'ori_arr.npy'))
pos_joi_arr = np.load(os.path.join(path_dataset, 'pos_joi_arr.npy'))
bcpos_arr = np.load(os.path.join(path_dataset, 'bcpos_arr.npy'))
index = np.load(os.path.join(path_dataset, 'index.npy'))

toc = clock()
logging.info('Done in %f sec.', toc-tic)

# Preprocess
data_mean = np.mean(bcpos_arr, axis=0)
data_std = np.std(bcpos_arr, axis=0)
bcpos_arr_ = (bcpos_arr - data_mean) / data_std

pos_joi_mean = np.mean(pos_joi_arr, axis=0)
pos_joi_std = np.std(pos_joi_arr, axis=0)
pos_joi_arr_ = (pos_joi_arr - pos_joi_mean) / pos_joi_std 

datain = np.concatenate(
            [bcpos_arr_[index[i, 0]:index[i, 1]-1, :]\
            for i in range(index.shape[0])],
            axis=0).astype('float32')
dataout = np.concatenate(
            [bcpos_arr_[index[i, 0]+1:index[i, 1], :]\
            for i in range(index.shape[0])],
            axis=0).astype('float32')
dataz = np.concatenate(
            [pos_joi_arr_[index[i, 0]+1:index[i, 1], :]\
            for i in range(index.shape[0])],
            axis=0).astype('float32')
data_index = np.concatenate(
                [[[index[i, 0]-i, index[i, 1]-i-1]]\
                for i in range(index.shape[0])],
                axis=0)

# shuffle
np.random.shuffle(data_index)

# split the dataset
n_seq_train = data_index.shape[0] * 8 / 10
index_train = data_index[:n_seq_train].copy()
index_test = data_index[n_seq_train:].copy()

# numper of frames in each dataset
cnt_frames_train = np.sum([index_train[i, 1] - index_train[i, 0] 
                            for i in range(index_train.shape[0])])
cnt_frames_test = np.sum([index_test[i, 1] - index_test[i, 0] 
                            for i in range(index_test.shape[0])])

# saving the data files
np.save('data_index', data_index)  
np.save('index_train', index_train)
np.save('index_test', index_test)
np.save('datain', datain)
np.save('dataout', dataout)
np.save('data_mean', data_mean)
np.save('data_std', data_std)
np.save('dataz', dataz)
np.save('pos_joi_mean', pos_joi_mean)
np.save('pos_joi_std', pos_joi_std)

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

toc = clock()
logging.info('Done in %f sec.', toc-tic)

# Direct Loss (for comparison)
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

# Training 
prev_loss_train = float('inf')
loss_hist_train = []
loss_hist_test = []

epoch = 0
# while 1:
while epoch <= 5000:
    tic = clock()

    # shuffle
    index_train_ = index_train.copy()
    np.random.shuffle(index_train_) 

    # train phase
    list_loss = []
    cnt_frames = 0
    for i in range(index_train_.shape[0]): 
        start = index_train_[i, 0]
        end = index_train_[i, 1]
        din = datain[start:end, :]
        dout = dataout[start:end, :]
        dz = dataz[start:end, :]
        loss = model.train(din, dz, dout)
        list_loss.append(loss * dout.shape[0])
    loss_train = np.sqrt(np.sum(list_loss) / cnt_frames_train)
    loss_hist_train.append([epoch, loss_train])

    toc = clock()
    logging.info('epoch: %d\tloss_train: %f\tlearning_rate: %.2f\ttime: %f',
                    epoch, loss_train, model.lr.get_value(), toc-tic)

    # learning rate decay
    if en_decay and prev_loss_train - loss_train <= epsl_decay:
        lr /= 2

    # test phase
    if en_test and (epoch+1) % test_cycle == 0:
        list_loss = []
        for i in range(index_test.shape[0]): 
            start = index_test[i, 0]
            end = index_test[i, 1]
            din = datain[start:end, :]
            dout = dataout[start:end, :]
            dz = dataz[start:end, :]
            pred = model.predict(din, dz)
            loss = np.mean((pred - dout) ** 2)           
            list_loss.append(loss * dout.shape[0])
        loss_test = np.sqrt(np.sum(list_loss) / cnt_frames_test)
        loss_hist_test.append([epoch, loss_test])
        logging.info('loss_test: %f', loss_test)

    # wrap up
    prev_loss_train = loss_train
    model.save(os.path.join(path_models, 'model'))
    np.save(os.path.join(path_models, 'loss_hist_train'), 
            np.array(loss_hist_train))
    np.save(os.path.join(path_models, 'loss_hist_test'), 
            np.array(loss_hist_test))
    epoch += 1

