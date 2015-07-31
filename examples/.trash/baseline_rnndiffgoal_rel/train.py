import sys
sys.path.append('/deep/u/kuanfang/action-prediction/');

import random
from time import clock

import numpy
from scipy import misc
from theano.tensor.shared_randomstreams import RandomStreams
numpy_rng  = numpy.random.RandomState(1)
theano_rng = RandomStreams(1)

from utils.io import *
from utils.log import *

from hyperparams import *
from rnn.rnndiffgoall1 import * 


# log 
################################################################################
logInfo = LogInfo('LOG.txt')


# load data
################################################################################
tic = clock()

subject = read_subject(path_data)
# print_subject(subject)

toc = clock()
logInfo.mark('time of loading data: ' + str(toc - tic))


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
        pos = activity['pos_rel']
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



# initialization
################################################################################
print 'start initialization ...'
tic = clock()

model = RNNL1(dimx, dimy, dimh)

if not os.path.exists(path_models):
    os.makedirs(path_models)

toc = clock()
logInfo.mark('time of initializing the model: ' + str(toc - tic))

print '... done'


# training
################################################################################
epoch = 0
prev_cost_train = float('inf')
while 1:
# while epoch <= 5:
    tic = clock()

    cost_train = 0
    num_seq = 0

    activity_labels = subject.keys()
    random.shuffle(activity_labels)
    for activity_label in activity_labels:
        # print '------------------------ <Directory> ------------------------'
        # print 'activity_label: {}'.format(activity_label)
        directory = subject[activity_label]
        activities = directory['activities']
        labeling = directory['labeling']

        for id in activities.keys():
            # print '\t--------------- <Activity> ---------------'
            activity = activities[id]
            sub_activities = labeling[id]
            # print '\tactivity_id: {}'.format(activity['activity_id'])

            activity_id = activity['activity_id']
            objects = activity['objects']
            # ori = activity['ori']
            # ori_conf = activity['ori_conf']
            pos = activity['pos_rel']
            pos_conf = activity['pos_conf']

            random.shuffle(sub_activities)
            for sub_activity in sub_activities:
                # print '\t\t----- <Sub-activity> -----'
                sub_activity_id = sub_activity['sub_activity_id']
                start_frame = sub_activity['start_frame']   # 0-based
                end_frame = sub_activity['end_frame']       # 0-based
                # affordances = sub_activity['affordances']

                cost_train += model.train(pos[start_frame:end_frame+1-1, :],
                                            pos[end_frame, :],
                                            pos[start_frame+1:end_frame+1, :], 
                                            lr)

                num_seq += 1

    toc = clock()

    cost_train = numpy.sqrt(cost_train / num_seq)   # RMSD
    logInfo.mark('# epoch: {}\
        \tcost_train: {}\tlearning_rate: {}\ttime: {}'\
        .format(epoch, cost_train, lr, toc-tic))

    # learning rate decay
    en_decay = False
    if en_decay and prev_cost_train - cost_train <= epsl_decay:
        lr /= 2
        print 'learning rate decay to {}. '.format(lr)

    prev_cost_train = cost_train

    model.save(os.path.join(path_models, 'model'))

    epoch += 1
