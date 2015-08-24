import os
import numpy as np
import theano
import theano.tensor as T
import logging
from time import clock
import random
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
from model.rnn import *
import util.my_logging
from util.mocap_utils import *
import pdb
import argparse


# Compilation Configuration
# theano.config.mode = 'FAST_COMPILE'
theano.config.mode = 'FAST_RUN'
# theano.config.mode = 'DEBUG_MODE'
# theano.config.exception_verbosity = 'high'

parser = argparse.ArgumentParser(description='Train on the MIT dataset using LSTMs.')
parser.add_argument('--inputs', type=str, required=True, nargs='+', dest='inputs')
parser.add_argument('--output', type=str, required=True, dest='output', choices=['frame', 'joi'])
parser.add_argument('--n_hidden', type=int, dest='n_hidden', default=128, help='Size of hidden layer')
parser.add_argument('--epochs', type=int, dest='num_epochs', help='Number of epochs')
parser.add_argument('--joi', type=int, dest='joi', nargs='+', default=None, help='Choices of joints of interest')
parser.add_argument('--activation', type=str, dest='activation_type', default='tanh', help='Activation function')
parser.add_argument('--batch_size', type=int, dest='bs', default=100, help='Batch size')
parser.add_argument('--learning_rate', type=float, dest='lr', default=1e-1, help='Learning rate')
parser.add_argument('--percent_train', type=float, dest='percent_train', default=0.8, help='Ratio of train to test')
parser.add_argument('--num_samples', dest='num_samples', default='all', help='Number of samples to work with. "All" to use all')
parser.add_argument('--final_frame_lookahead', type=int, default=100, help='Number of frames to lookahead for goal joi state')
parser.add_argument('--load_data', type=int, dest='load_saved_data', default=0) 
parser.add_argument('--load_model', type=int, dest='load_saved_model', default=0)
parser.add_argument('--output_subdirectory', type=str, dest='output_subdirectory', help='Subdirectory in the outputs path to save or load contents from', required=True)
parser.add_argument('--en_test', type=int, dest='en_test', default=1, help='Enable testing during the training cycles')
parser.add_argument('--test_cycle', type=int, dest='test_cycle', default=10)
parser.add_argument('--en_decay', type=int, dest='en_decay', default=0)
parser.add_argument('--epsl_decay', type=float, dest='epsl_decay', default=1e-3)
parser.add_argument('--print_console', type=int, dest='print_console', default=1)
parser.add_argument('--logfile', type=str, dest='logfile', default='logfile.log')
parser.add_argument('--activity_file', type=str, dest='activity_file', default="")
parser.add_argument('--optimize_method', type=str, dest='optimize_method', default='adadelta')

args = parser.parse_args()

inputs = args.inputs
output = args.output
n_h = args.n_hidden
num_epochs = args.num_epochs
joi = args.joi
bs = args.bs
lr = args.lr
percent_train = args.percent_train
num_samples = int(args.num_samples) if (args.num_samples != 'all' and args.num_samples != 'All' and args.num_samples != 'ALL') else None
final_frame_lookahead = args.final_frame_lookahead
load_saved_data = True if args.load_saved_data else False
load_saved_model = True if args.load_saved_model else False
path_outputs = '../outputs/%s/' % args.output_subdirectory
path_dataset = '../data/mit'
optimize_method = args.optimize_method
en_test = True if args.en_test else False
test_cycle = args.test_cycle
en_decay = True if args.en_decay else False
epsl_decay = args.epsl_decay
print_console = args.print_console
activation_type = args.activation_type
if not os.path.exists(path_outputs):
	os.mkdir(path_outputs)
if print_console:
	logfile = sys.stdout
else:
	logfile = os.path.join(path_outputs, args.logfile)
	logfile = open(logfile, 'w')

print >>logfile, "Parameter Information:"
print >>logfile, "Inputs:", inputs
print >>logfile, "Output: %s" % output
print >>logfile, "Hidden Layer Size: %d" % n_h
print >>logfile, "Number of Epochs: %d" % num_epochs
print >>logfile, "Batch size: %d" % bs
print >>logfile, "Learning rate: %f" % lr
print >>logfile, "Joints of Interest:", joi 
print >>logfile, "Ratio of training to testing samples: %f" % percent_train
print >>logfile, "Number of samples to train: " + str(num_samples)
print >>logfile, "Final frame lookahead: %d" % final_frame_lookahead
print >>logfile, "Activation type: %s" % activation_type
print >>logfile, "Output subdirectory: %s" % path_outputs
print >>logfile, "Enable testing: %r" % en_test
print >>logfile, "Load saved data: %r" % load_saved_data
print >>logfile, "Load saved model: %r" % load_saved_model
print >>logfile, ""

from util.mit_utils import *
from util.motion import load_data

# Load data
print >>logfile, "Building data..."

if not load_saved_data or not os.path.exists(path_outputs) or not os.path.exists(os.path.join(path_outputs, 'datain.npy')):
	print >>logfile, "Loading data from motion.mat"
	motion_file = '../data/mit/motion.mat'
	if args.activity_file:
		x, seqlen, data_mean, data_std, offsets = preprocess(1, args.activity_file)	
	else:
		x, seqlen, data_mean, data_std, offsets = load_data(motion_file, shared=False)

	x_t = x
	z_t = get_joi(x_t, joi, data_mean, data_std, offsets)
	z_T = final_frame(z_t, final_frame_lookahead)

	print >>logfile, "x_t shape", x_t.shape
	print >>logfile, "z_t shape", z_t.shape
	print >>logfile, "z_T shape", z_T.shape
	print >>logfile, ""

	datain = []

	# Initialize model
	print >>logfile, "Initializing model..."
	for input in inputs:
		if input == 'frame':
			datain.append(x_t[:-1])
		if input == 'joi_current':
			datain.append(z_t[:-1])
		if input == 'joi_next':
			datain.append(z_t[1:])
		if input == 'joi_final':
			datain.append(z_T[:-1])
	
	if output == 'frame':
		dataout = x_t[1:]
	elif output == 'joi':
		dataout = z_t[1:]

	if not os.path.exists(path_outputs):
		os.makedirs(path_outputs)
	np.save(os.path.join(path_outputs, 'data_mean.npy'), data_mean)
	np.save(os.path.join(path_outputs, 'data_std.npy'), data_std)
	np.save(os.path.join(path_outputs, 'offsets.npy'), offsets)
	#np.save(os.path.join(path_outputs, 'datain.npy'), datain)
	np.save(os.path.join(path_outputs, 'dataout.npy'), dataout)
else:
	print >>logfile, "Loading existing numpy data"
	datain = np.load(os.path.join(path_outputs,'datain.npy'))
	dataout = np.load(os.path.join(path_outputs,'dataout.npy'))
	data_mean = np.load(os.path.join(path_outputs,'data_mean.npy'))
	data_std = np.load(os.path.join(path_outputs,'data_std.npy'))
	offsets = np.load(os.path.join(path_outputs,'offsets.npy'))

data_index = np.array(range(len(datain)))
np.random.shuffle(data_index)
data_index = range(num_samples) if (num_samples and num_samples <= len(datain[0])) else range(len(datain[0]))
data_index = np.array(data_index)
np.random.shuffle(data_index)
n_seq = len(data_index)
index_train = data_index[:int(percent_train*n_seq)]
index_test = data_index[int(percent_train*n_seq):n_seq]

cnt_frames_train = index_train.shape[0]
cnt_frames_test = index_test.shape[0]

n_x = sum(data.shape[1] for data in datain)
n_y = dataout.shape[1]
n_h = n_h

print >>logfile, "Input data shape", [data.shape for data in datain]
print >>logfile, "Output data shape", dataout.shape
print >>logfile, "Number of train", cnt_frames_train
print >>logfile, "Number of test", cnt_frames_test
print >>logfile, ""
print >>logfile, "Building network..."
if activation_type == 'tanh':
	activation = T.tanh
elif activation_type == 'relu':
	activation = lambda x: T.switch(x<0, 0, x)

relu = lambda x: T.switch(x<0, 0, x)

cells = StackedCells(n_x, layers=[n_h], activation=activation, celltype=LSTM)
cells.layers.append(Layer(n_h, n_y, relu))

#cells.layers.insert(0, Layer(n_fc1, n_fc2, acti_fc))
#cells.layers.insert(0, Layer(n_x, n_fc1, acti_fc))


if len(datain) == 1: RNN = RNN1L
elif len(datain) == 2: RNN = RNN1LZ
elif len(datain) == 3: RNN = RNN1LZZ
model = RNN(cells, dynamics=lambda x, y: x+y, optimize_method=optimize_method)
if load_saved_model:
	model.load(os.path.join(path_outputs, 'model'))

print >>logfile, "Training model..."
prev_loss_train = float('inf')
loss_hist_train = []
loss_hist_test = []

epoch = 0
list_losses = []
while epoch <= num_epochs:
	index_train_ = index_train.copy()
	np.random.shuffle(index_train_)
	# train phase
	list_loss = []
	for i in range(index_train_.shape[0] / bs):
		data = index_train_[i*bs:(i+1)*bs]
		din = [datain_[data, :] for datain_ in datain]
		dout = dataout[data, :]
		loss = model.train(dout, *din)
		list_loss.append(loss * dout.shape[0])
	loss_train = np.sqrt(np.sum(list_loss) / cnt_frames_train)
	list_losses.append(list_loss)
	loss_hist_train.append([epoch, loss_train])

	print >>logfile, 'epoch: %d\tloss_train: %f\tlearning_rate: %.2f' % (epoch, loss_train, model.lr.get_value())
	if en_decay and prev_loss_train - loss_train <= epsl_decay:
		lr /= 2

    # test phase
	if en_test and (epoch+1) % test_cycle == 0:
		list_loss = []
		for i in range(index_test.shape[0] / bs): 
			data = index_test[i*bs:(i+1)*bs]
			din = [datain_[data, :] for datain_ in datain]
			dout = dataout[data, :]
			pred = model.predict(*din)
			loss = np.mean((pred - dout) ** 2)            
			list_loss.append(loss)
		loss_test = np.sqrt(np.mean(list_loss))
		loss_hist_test.append([epoch, loss_train])
		print >>logfile, 'loss_test: %f' % loss_test

	prev_loss_train = loss_train
	model.save(os.path.join(path_outputs, 'model'))
	np.save(os.path.join(path_outputs, 'loss_hist_train'), 
			np.array(loss_hist_train))
	np.save(os.path.join(path_outputs, 'loss_hist_test'), 
			np.array(loss_hist_test))
	epoch += 1

print >>logfile, "Finished training..."
print >>logfile, "Printing loss curves..."
plt.subplot(2, 1, 1)
plt.plot(map(lambda x: x[0], loss_hist_train), map(lambda x: x[1], loss_hist_train))
plt.title('Training loss history')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(os.path.join(path_outputs, 'training_loss.png'), bbox_inches='tight')
plt.close()

plt.subplot(2, 1, 1)
plt.plot(map(lambda x: x[0], loss_hist_test), map(lambda x: x[1], loss_hist_test))
plt.title('Testing loss history')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(os.path.join(path_outputs, 'testing_loss.png'), bbox_inches='tight')
plt.close()

print >>logfile, "Finished successfully."
logfile.close()
