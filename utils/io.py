# IO functions for cornell's CAD-120 dataset 

import os
import numpy
import numpy.random
from scipy import misc

from params_cad120 import *

# class SubActivity(object):
# 
#     def __init__(self, id, act_id, obj_id, obj_type):
#         self.id = id


def read_activity_labels(filename):
    """
    Read the activityLabels.txt file. 
    """
    activity_labels = []
    for line in open(filename):
        words = line.split(',')
        activity_labels.append(words[0])
    return activity_labels


def read_skeleton_data(filename):
    """
    Read a sequence of skeleton movements. 
    """
    len_seq = sum(1 for line in open(filename))

    ori = numpy.zeros((len_seq, 9*11))
    pos = numpy.zeros((len_seq, 3*15))
    ori_conf = numpy.zeros((len_seq, 11))
    pos_conf = numpy.zeros((len_seq, 15))

    row = 0
    for line in open(filename):
        words = line.split(',')

        if len(words) != 1 + 9*11+3*15+11+15 + 1:
            break

        vals = [float(words[i]) for i in range(1 + 9*11+3*15+11+15)]

        id = float(words[0])
        ori[row, :] = numpy.array([vals[idx] for idx in idx_ori])
        pos[row, :] = numpy.array([vals[idx] for idx in idx_pos])
        ori_conf[row, :] = numpy.array([vals[idx] for idx in idx_ori_conf])
        pos_conf[row, :] = numpy.array([vals[idx] for idx in idx_pos_conf])

        row += 1

    return ori, ori_conf, pos, pos_conf


def read_activity(path):
    """
    Read the activity sequences of skeleton movements in a activity pathectory. 
    """
    path_label = os.path.join(path, 'activityLabel.txt')

    activity_list = []

    for line in open(path_label):
        activity = {}

        words = line.split(',')
        id = int(words[0])
        activity_id = words[1]
        subject_id = words[2]
        activity['id'] = id
        activity['activity_id'] = activity_id

        objects = []
        for i in range(3, len(words)-1):
            """
            obj_info = words[i].split(':')
            object_id = obj_info[0]
            object_type = obj_info[1]
            objects.append(object_type)
            """
            object_id, object_type = words[i].split(':')
            objects.append(object_type)
        activity['objects'] = objects

        path_subact = os.path.join(path, words[0] + '.txt')
        ori, ori_conf, pos, pos_conf = read_skeleton_data(path_subact)
        activity['ori'] = ori
        activity['ori_conf'] = ori_conf
        activity['pos'] = pos
        activity['pos_conf'] = pos_conf

        activity_list.append(activity)

    return activity_list
            

def read_subject(path):
    """
    Read all activity sequences of skeleton movements in a Subject folder.
    """
    subject = {}

    for activity_label in os.listdir(path):
        path_activity = os.path.join(path, activity_label)
        if os.path.isdir(path_activity):
            activity = read_activity(path_activity)             
            subject[activity_label] = activity 

    return subject


def print_activity_list(activity_list):
    """
    Print the activity list. 
    """
    for activity in activity_list:
        print '------------------------------'
        print 'id: {}'.format(activity['id'])
        print 'activity_id: {}'.format(activity['activity_id'])
        print 'objects: {}'.format(activity['objects'])

        print 'seq_len: {}'.format(activity['ori'].shape[0])
        # print 'ori shape: {}'.format(activity['ori'].shape)
        # print 'ori_conf shape: {}'.format(activity['ori_conf'].shape)
        # print 'pos shape: {}'.format(activity['pos'].shape)
        # print 'pos_conf shape: {}'.format(activity['pos_conf'].shape)


def print_subject(subject):
    """
    Print the subject. 
    """
    for activity_label, activity_list in subject.iteritems():
        print '############################################################'
        print 'activity_label: {}'.format(activity_label)
        print_activity_list(activity_list)
