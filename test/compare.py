import sys
sys.path.append('/deep/u/kuanfang/action-prediction');

import random

import numpy
from scipy import misc
from time import clock
from theano.tensor.shared_randomstreams import RandomStreams
numpy_rng  = numpy.random.RandomState(1)
theano_rng = RandomStreams(1)

from utils.io import *
from utils.log import *

from rnn.rnnl1 import * 

path_data = '../CAD-120/Subject3_annotations'

# parameter
################################################################################
dimx = len(idx_pos)
dimy = len(idx_pos)
dimh = 20

lr = 1e-1
bs = 1

epsl_decay = 2.

path_models = 'models/'


# log 
################################################################################


# load data
################################################################################
tic = clock()

subject = read_subject(path_data)
# print_subject(subject)

toc = clock()
print 'time of loading data: ' + str(toc - tic)


# RMSD to 0
################################################################################
S = 0
num_row = 0
num_col = dimx
activity_labels = subject.keys()
for activity_label in activity_labels:
    directory = subject[activity_label]
    activities = directory['activities']
    labeling = directory['labeling']

    for id in activities.keys():
        activity = activities[id]
        sub_activities = labeling[id]

        activity_id = activity['activity_id']
        objects = activity['objects']
        # ori = activity['ori']
        # ori_conf = activity['ori_conf']
        pos = activity['pos']
        pos_conf = activity['pos_conf']
        for sub_activity in sub_activities:
            # print '\t\t----- <Sub-activity> -----'
            sub_activity_id = sub_activity['sub_activity_id']
            start_frame = sub_activity['start_frame']   # 0-based
            end_frame = sub_activity['end_frame']       # 0-based
            # affordances = sub_activity['affordances']

            S += numpy.sum((pos[start_frame:end_frame+1-1, :] -
                             pos[start_frame+1:end_frame+1, :]) ** 2)
            num_row += end_frame - start_frame

S = numpy.sqrt(S / num_row / num_col)
print 'num_row: {}'.format(num_row)
print 'num_col: {}'.format(num_col)
print 'S: {}'.format(S)

