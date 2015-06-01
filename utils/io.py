# IO functions for cornell's CAD-120 dataset 

import os
import numpy
import numpy.random
from scipy import misc

from params_cad120 import *
from rotation import * 


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

        # pos_rel = world_to_relative_pos(pos)
        # activity['pos_rel'] = pos_rel

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

            # print '\tseq_len: {}'.format(activity['pos_rel'])

            sub_activities = labeling[id]
            for sub_activity in sub_activities:
                print '\t\t----- <Sub-activity> -----'
                print '\t\tsub_activity_id: {}'.format(
                        sub_activity['sub_activity_id'])
                print '\t\tstart_frame: {}'.format(sub_activity['start_frame'])
                print '\t\tend_frame: {}'.format(sub_activity['end_frame'])
                print '\t\taffordances: {}'.format(sub_activity['affordances'])



"""
Relative Position
"""

def world_to_relative_pos(pos):
    """
    Convert world position array pos into relative position array. 

    Parameters
    ----------
    pos: numpy array
        World position array. 

    Returns
    -------
    pos_rel: numpy array
        Relative position array. 
    """
    pos_rel = pos.copy()

    for p, c in reversed(connect):
        pos_rel[:, c*dim_pos+0] -= pos_rel[:, p*dim_pos+0]
        pos_rel[:, c*dim_pos+1] -= pos_rel[:, p*dim_pos+1]
        pos_rel[:, c*dim_pos+2] -= pos_rel[:, p*dim_pos+2]

    return pos_rel


def relative_to_world_pos(pos_rel):
    """
    Convert relative position array pos into world position array. 

    Parameters
    ----------
    pos_rel: numpy array
        Relative position array. 

    Returns
    -------
    pos: numpy array
        World position array. 
    """
    pos = pos_rel.copy()

    for p, c in connect:
        pos[:, c*dim_pos+0] += pos[:, p*dim_pos+0]
        pos[:, c*dim_pos+1] += pos[:, p*dim_pos+1]
        pos[:, c*dim_pos+2] += pos[:, p*dim_pos+2]

    return pos


def subject_get_relative_position(subject):
    """
    Compute relative position array for each activity in subject. 
    """
    # for activity_label in subject.keys():
    #     for id in directory[activity_label]['activities'].keys():
    #         pos = directory[activity_label]['activities'][id]['pos']
    #         pos_rel = world_to_relative_pos(activity['pos'])
    #         directory[activity_label]['activities'][id]['pos_rel'] = pos_rel

    for activity_label, directory in subject.iteritems():
        activities = directory['activities']
        labeling = directory['labeling']

        for id in activities.keys():
            activity = activities[id]
            pos_rel = world_to_relative_pos(activity['pos'])

            activity['pos_rel'] = pos_rel

    return subject


def subject_get_relative_position(subject):
    """
    Compute relative position array for each activity in subject. 
    """
    # for activity_label in subject.keys():
    #     for id in directory[activity_label]['activities'].keys():
    #         pos = directory[activity_label]['activities'][id]['pos']
    #         pos_rel = world_to_relative_pos(activity['pos'])
    #         directory[activity_label]['activities'][id]['pos_rel'] = pos_rel

    for activity_label, directory in subject.iteritems():
        activities = directory['activities']
        labeling = directory['labeling']

        for id in activities.keys():
            activity = activities[id]
            pos_rel = world_to_relative_pos(activity['pos'])

            activity['pos_rel'] = pos_rel

    return subject


def vec2rmat(a):
    """
    Convert a 9-dim vector to a 3x3 rotation matrix.  
    """
    return numpy.reshape(a, (3, 3))


def pos2data(pos, ori):
    """
    """
    # root = joint_idx['torso']

    data = numpy.zeros((pos.shape[0], 3+len_pos))

    p_o = [None] * (num_pos-1)

    for row in range(pos.shape[0]):
        rmat = numpy.reshape(ori[row, root*dim_ori+0:(root+1)*dim_ori], 
                            (3, 3))
        v_root = rmat_to_r3(rmat)
        p_root = pos[row, root*dim_pos:(root+1)*dim_pos]

        cnt = 0;
        for j in range(num_pos):
            if j != root:
                """
                pos_w(x) = pos_w(o) + rmat_w(o) * pos_o(x)
                pos_w(o) = rmat_w(o)^T * (pos_w(x) - pos_w(o))
                """
                p_w = pos[row, j*dim_pos:(j+1)*dim_pos]
                p_o[cnt] = numpy.dot(rmat.T, p_w - p_root)
                cnt += 1

        data[row, :] =  numpy.concatenate([v_root, p_root] + p_o, axis=1)

    return data



def data2pos(data):
    """
    """
    # root = joint_idx['torso']
    pos = numpy.zeros((data.shape[0], len_pos))

    for row in range(data.shape[0]):
        v_root = data[row, 0:3]
        p_root = data[row, 3:6]

        rmat = r3_to_rmat(v_root)

        cnt = 0;
        for j in range(num_pos):
            if j == root:
                p_w = p_root 
            else:
                p_o = data[row, 6+cnt*dim_pos:6+(cnt+1)*dim_pos]
                p_w = p_root + numpy.dot(rmat, p_o)
                cnt += 1
            pos[row, j*dim_pos:(j+1)*dim_pos] = p_w

    return pos


# """
# Relative Orientation
# """
# 
# def world_to_relative_ori(ori):
#     """
#     Convert world orientaion array ori into torsor-centric 
#     relative orientation array. 
# 
#     Parameters
#     ----------
#     ori: numpy array
#         World orientaion array. 
# 
#     Returns
#     -------
#     ori_rel: numpy array
#         Relative orientation array. 
#     """
#     ori_rel = ori.copy()
# 
#     for p, c in reversed(connect):
#         ori_rel[:, c*dim_ori+0] -= ori_rel[:, p*dim_ori+0]
#         ori_rel[:, c*dim_ori+1] -= ori_rel[:, p*dim_ori+1]
#         ori_rel[:, c*dim_ori+2] -= ori_rel[:, p*dim_ori+2]
# 
#     return ori_rel
