# IO functions for cornell's CAD-120 dataset 

import os
import numpy
import numpy.random
from scipy import misc

from params_cad120 import *


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
    len_seq = sum(1 for line in open(filename))-1

    ori = numpy.zeros((len_seq, 9*11), dtype = numpy.float32)
    pos = numpy.zeros((len_seq, 3*15), dtype = numpy.float32)
    ori_conf = numpy.zeros((len_seq, 11), dtype = numpy.float32)
    pos_conf = numpy.zeros((len_seq, 15), dtype = numpy.float32)

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


def read_activities(path_activity):
    """
    Read the activity sequences of skeleton movements in a activity pathectory. 
    """
    path_label = os.path.join(path_activity, 'activityLabel.txt')

    activities = {}

    for line in open(path_label):
        activity = {}

        words = line.split(',')

        id = int(words[0])
        activity_id = words[1]
        subject_id = words[2]
        activity['activity_id'] = activity_id

        objects = []
        for i in range(3, len(words)-1):
            object_id, object_type = words[i].strip().split(':')
            objects.append(object_type)
        activity['objects'] = objects

        path_subact = os.path.join(path_activity, words[0] + '.txt')
        ori, ori_conf, pos, pos_conf = read_skeleton_data(path_subact)
        activity['ori'] = ori
        activity['ori_conf'] = ori_conf
        activity['pos'] = pos
        activity['pos_conf'] = pos_conf

        pos_rel = world_to_relative_pos(pos)
        activity['pos_rel'] = pos_rel

        activities[id] = activity

    return activities
            

def read_labeling(path_activity):
    """
    Read labeling.txt. 
    """
    path_labeling = os.path.join(path_activity, 'labeling.txt')

    labeling = {}

    for line in open(path_labeling):
        sub_activity = {}

        words = line.strip('\n').split(',')
        id = int(words[0])
        start_frame = int(words[1])
        end_frame = int(words[2])
        sub_activity_id = words[3]

        affordances = []
        for i in range(4, len(words)):
                affordance_id = words[i]
                affordances.append(affordance_id)

        sub_activity['start_frame'] = start_frame - 1   # 1-based to 0-based
        sub_activity['end_frame'] = end_frame - 1       # 1-based to 0-based
        sub_activity['sub_activity_id'] = sub_activity_id
        sub_activity['affordances'] = affordances

        if not labeling.has_key(id):
            labeling[id] = []

        labeling[id].append(sub_activity)

    return labeling


def read_subject(path_subject):
    """
    Read all activity sequences of skeleton movements in a Subject folder.
    """
    subject = {}

    for activity_label in os.listdir(path_subject):
        path_activity = os.path.join(path_subject, activity_label)
        if os.path.isdir(path_activity):
            activities = read_activities(path_activity)             
            labeling = read_labeling(path_activity)             

            directory = {}
            directory['activities'] = activities
            directory['labeling'] = labeling

            subject[activity_label] = directory

    return subject
    

def print_activities(activities):
    """
    Print the activities on the screen. 
    """
    for id, activity in activities.iteritems():
        print '------------------------------'
        print 'id: {}'.format(id)
        print 'activity_id: {}'.format(activity['activity_id'])
        print 'objects: {}'.format(activity['objects'])

        print 'seq_len: {}'.format(activity['ori'].shape[0])


def print_labeling(labeling):
    """
    Print the labeling on the screen. 
    """
    raise Exception('Not implemented yet. ')


def print_subject(subject):
    """
    Print the subject on the screen. 
    """

    for activity_label, directory in subject.iteritems():
        print '------------------------- <Directory> -------------------------'
        print 'activity_label: {}'.format(activity_label)
        activities = directory['activities']
        labeling = directory['labeling']

        for id in activities.keys():
            print '\t--------------- <Activity> ---------------'
            print '\tid: {}'.format(id)

            activity = activities[id]
            print '\tactivity_id: {}'.format(activity['activity_id'])
            print '\tobjects: {}'.format(activity['objects'])
            print '\tseq_len: {}'.format(activity['ori'].shape[0])

            sub_activities = labeling[id]
            for sub_activity in sub_activities:
                print '\t\t----- <Sub-activity> -----'
                print '\t\tsub_activity_id: {}'.format(
                        sub_activity['sub_activity_id'])
                print '\t\tstart_frame: {}'.format(sub_activity['start_frame'])
                print '\t\tend_frame: {}'.format(sub_activity['end_frame'])
                print '\t\taffordances: {}'.format(sub_activity['affordances'])


def world_to_relative_pos(pos):
    """

    """
    pos_rel = pos.copy()

    for p, c in reversed(connect):
        pos_rel[:, c*dim_pos+0] -= pos_rel[:, p*dim_pos+0]
        pos_rel[:, c*dim_pos+1] -= pos_rel[:, p*dim_pos+1]
        pos_rel[:, c*dim_pos+2] -= pos_rel[:, p*dim_pos+2]

    return pos_rel


def relative_to_world_pos(pos_rel):
    """

    """
    pos = pos_rel.copy()

    for p, c in connect:
        pos[:, c*dim_pos+0] += pos[:, p*dim_pos+0]
        pos[:, c*dim_pos+1] += pos[:, p*dim_pos+1]
        pos[:, c*dim_pos+2] += pos[:, p*dim_pos+2]

    return pos

# def world_to_relative_pos(pos):
#     """
# 
#     """
#     pos_rel = numpy.zeros(pos.shape)
# 
#     for p, c in reversed(connect):
#         pos_rel[:, c*dim_pos+0] = pos[:, c*dim_pos+0] - pos[:, p*dim_pos+0]
#         pos_rel[:, c*dim_pos+1] = pos[:, c*dim_pos+1] - pos[:, p*dim_pos+1]
#         pos_rel[:, c*dim_pos+2] = pos[:, c*dim_pos+2] - pos[:, p*dim_pos+2]
# 
#     root = connect[0][0]
#     pos_rel[:, root*dim_pos+0] = pos[:, root*dim_pos+0]
#     pos_rel[:, root*dim_pos+1] = pos[:, root*dim_pos+1]
#     pos_rel[:, root*dim_pos+2] = pos[:, root*dim_pos+2]
# 
#     return pos_rel
# 
# 
# def relative_to_world_pos(pos_rel):
#     """
# 
#     """
#     pos = numpy.zeros(pos_rel.shape)
# 
#     root = connect[0][0]
#     pos[:, root*dim_pos+0] = pos_rel[:, root*dim_pos+0]
#     pos[:, root*dim_pos+1] = pos_rel[:, root*dim_pos+1]
#     pos[:, root*dim_pos+2] = pos_rel[:, root*dim_pos+2]
# 
#     for p, c in connect:
#         pos[:, c*dim_pos+0] = pos_rel[:, c*dim_pos+0] + pos_rel[:, p*dim_pos+0]
#         pos[:, c*dim_pos+1] = pos_rel[:, c*dim_pos+1] + pos_rel[:, p*dim_pos+1]
#         pos[:, c*dim_pos+2] = pos_rel[:, c*dim_pos+2] + pos_rel[:, p*dim_pos+2]
# 
#     return pos

# def world_to_relative_pos(subject):
#     """
#     Print the subject on the screen. 
#     """
# 
#     for activity_label, directory in subject.iteritems():
#         activities = directory['activities']
#         labeling = directory['labeling']
# 
#         for id in activities.keys():
#             activity = activities[id]
#             pos = activity['pos']
# 
#             for i in range(pos.shape[0]):
#                 root = joint_idx['torsor']
#                 for j in range(num_pos):
#                     if j != root:
#                         pos[i, j*dim_pos+0] -= pos[i, root*dim_pos+0]
#                         pos[i, j*dim_pos+1] -= pos[i, root*dim_pos+1]
#                         pos[i, j*dim_pos+2] -= pos[i, root*dim_pos+2]
# 
#             subject[activity_label]['activities'][id]['relative_pos'] = pos
# 

