import os
import numpy as np
from collections import OrderedDict 

################################################################################
# Skeleton Parameters
################################################################################
skel = {}

# Names of Joints
joints = [
            'torso',
            'neck',
            'left_shoulder',
            'right_shoulder',
            'left_hip',
            'right_hip',

            'head',
            'left_elbow',
            'right_elbow',
            'left_knee',
            'right_knee',

            'left_hand',
            'right_hand',
            'left_foot',
            'right_foot'
            ]
skel['joints'] = joints

tree = []
# torso
tree.append({
            'parent': None,
            'children': [joints.index('neck'),
                        joints.index('left_shoulder'),
                        joints.index('right_shoulder'),
                        joints.index('left_hip'),
                        joints.index('right_hip')]
            })
# neck
tree.append({
            'parent': joints.index('torso'),
            'children': [joints.index('head')]
            })
# left_shoulder
tree.append({
            'parent': joints.index('torso'),
            'children': [joints.index('left_elbow')]
            })
# right_shoulder
tree.append({
            'parent': joints.index('torso'),
            'children': [joints.index('right_elbow')]
            })
# left_hip
tree.append({
            'parent': joints.index('torso'),
            'children': [joints.index('left_knee')]
            })
# right_hip
tree.append({
            'parent': joints.index('torso'),
            'children': [joints.index('right_knee')]
            })
# head
tree.append({
            'parent': joints.index('neck'),
            'children': []
            })
# left_elbow
tree.append({
            'parent': joints.index('left_shoulder'),
            'children': [joints.index('left_hand')]
            })
# right_elbow
tree.append({
            'parent': joints.index('right_shoulder'),
            'children': [joints.index('right_hand')]
            })
# left_knee
tree.append({
            'parent': joints.index('left_hip'),
            'children': [joints.index('left_foot')]
            })
# right_knee
tree.append({
            'parent': joints.index('right_hip'),
            'children': [joints.index('right_foot')],
            })
# left_hand
tree.append({
            'parent': joints.index('left_elbow'),
            'children': []
            })
# right_hand
tree.append({
            'parent': joints.index('right_elbow'),
            'children': []
            })
# left_foot
tree.append({
            'parent': joints.index('left_knee'),
            'children': []
            })
# right_foot
tree.append({
            'parent': joints.index('right_knee'),
            'children': []
            })

skel['tree'] = tree

connection = [(tree[c]['parent'], c) for c in range(1, len(joints))]
skel['connection'] = connection

################################################################################
# IO & Process Parameters
################################################################################
# Indices for reading data from file
ind_ori = [29, 30, 31, 32, 33, 34, 35, 36, 37, 15, 16, 17, 18, 19, 20, 21, 22, 
            23, 43, 44, 45, 46, 47, 48, 49, 50, 51, 71, 72, 73, 74, 75, 76, 77, 
            78, 79, 99, 100, 101, 102, 103, 104, 105, 106, 107, 127, 128, 129, 
            130, 131, 132, 133, 134, 135, 1, 2, 3, 4, 5, 6, 7, 8, 9, 57, 58, 59,
            60, 61, 62, 63, 64, 65, 85, 86, 87, 88, 89, 90, 91, 92, 93, 113, 
            114, 115, 116, 117, 118, 119, 120, 121, 141, 142, 143, 144, 145, 
            146, 147, 148, 149]
ind_pos = [39, 40, 41, 25, 26, 27, 53, 54, 55, 81, 82, 83, 109, 110, 111, 137, 
            138, 139, 11, 12, 13, 67, 68, 69, 95, 96, 97, 123, 124, 125, 151, 
            152, 153, 155, 156, 157, 159, 160, 161, 163, 164, 165, 167, 168,169]
ind_oriconf = [38, 24, 52, 80, 108, 136, 10, 66, 94, 122, 150]
ind_posconf = [42, 28, 56, 84, 112, 140, 14, 70, 98, 126, 154, 158, 162,166,170]

################################################################################
# Preprocess Functions (Read)
################################################################################
def read(path):
    """
    Read a subject
    """
    pos_arr, posconf_arr, ori_arr, oriconf_arr, subject \
        = merge(_read_subject(path))
    return pos_arr, posconf_arr, ori_arr, oriconf_arr, subject

def _read_activity_labels(filename):
    """
    Read the activityLabels.txt file. 
    """
    activity_labels = []
    for line in open(filename):
        words = line.split(',')
        activity_labels.append(words[0])
    return activity_labels

def _read_skeleton_data(filename):
    """
    Read a sequence of skeleton movements. 
    """
    len_arr = sum(1 for line in open(filename))-1

    ori_arr = np.zeros((len_arr, 9*11), dtype = np.float32)
    pos_arr = np.zeros((len_arr, 3*15), dtype = np.float32)
    oriconf_arr = np.zeros((len_arr, 11), dtype = np.float32)
    posconf_arr = np.zeros((len_arr, 15), dtype = np.float32)

    row = 0
    for line in open(filename):
        words = line.split(',')

        if len(words) != 1 + 9*11+3*15+11+15 + 1:
            break

        vals = [float(words[i]) for i in range(1 + 9*11+3*15+11+15)]

        id = float(words[0])
        ori_arr[row, :] = np.array([vals[ind] for ind in ind_ori])
        pos_arr[row, :] = np.array([vals[ind] for ind in ind_pos])
        oriconf_arr[row, :] = np.array([vals[ind] for ind in ind_oriconf])
        posconf_arr[row, :] = np.array([vals[ind] for ind in ind_posconf])

        row += 1

    return ori_arr, oriconf_arr, pos_arr, posconf_arr

def _read_activities(path_activity):
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
        ori_arr, oriconf_arr, pos_arr, posconf_arr \
            = _read_skeleton_data(path_subact)
        
        activity['ori_arr'] = ori_arr
        activity['oriconf_arr'] = oriconf_arr
        activity['pos_arr'] = pos_arr
        activity['posconf_arr'] = posconf_arr

        activities[id] = activity

    return activities

def _read_labeling(path_activity):
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
        sub_activity['end_frame'] = end_frame   # 1-based to 0-based
        sub_activity['sub_activity_id'] = sub_activity_id
        sub_activity['affordances'] = affordances

        if not labeling.has_key(id):
            labeling[id] = []

        labeling[id].append(sub_activity)

    return labeling

def _read_subject(path_subject):
    """
    Read all activity sequences of skeleton movements in a Subject folder.
    """
    subject = {}

    for activity_label in os.listdir(path_subject):
        path_activity = os.path.join(path_subject, activity_label)
        if os.path.isdir(path_activity):
            activities = _read_activities(path_activity)             
            labeling = _read_labeling(path_activity)             

            directory = {}
            directory['activities'] = activities
            directory['labeling'] = labeling

            subject[activity_label] = directory

    return subject

def merge(subject):
    """
    Merge the pos_arr's and ori_arr's into the same numpy array. 
    Modify the starting and endding frame numbers accordingly. 
    """
    subject_new = {}
    cnt = 0
    for activity_label, directory in subject.iteritems():
        activities = directory['activities']
        labeling = directory['labeling']
        activities_new = {}

        for id in activities.keys():
            activity = activities[id]

            activity_new = {}
            act_len = activity['ori_arr'].shape[0]
            activity_new['activity_id'] = activity['activity_id']
            activity_new['objects'] = activity['objects']
            activity_new['start_frame'] = cnt
            activity_new['end_frame'] = cnt + act_len - 1

            sub_activities_new = labeling[id]
            for sub_activity in sub_activities_new:
                sub_activity['start_frame'] += cnt
                sub_activity['end_frame'] += cnt

            activity_new['sub_activities'] =  sub_activities_new
            activities_new[id] = activity_new

            cnt += act_len
            
        subject_new[activity_label] = activities_new

    pos_arr = np.concatenate(\
            [activity['pos_arr']
                for activity_label, directory in subject.iteritems()
                for id, activity in directory['activities'].iteritems()
            ],
            axis=0)

    ori_arr = np.concatenate(\
            [activity['ori_arr']
                for activity_label, directory in subject.iteritems()
                for id, activity in directory['activities'].iteritems()
            ],
            axis=0)

    posconf_arr = np.concatenate(\
            [activity['posconf_arr']
                for activity_label, directory in subject.iteritems()
                for id, activity in directory['activities'].iteritems()
            ],
            axis=0)

    oriconf_arr = np.concatenate(\
            [activity['oriconf_arr']
                for activity_label, directory in subject.iteritems()
                for id, activity in directory['activities'].iteritems()
            ],
            axis=0)

    return pos_arr, posconf_arr, ori_arr, oriconf_arr, subject_new

def print_subject(subject):
    """
    Print the subject on the screen. 
    """

    for activity_label, activities in subject.iteritems():
        print '------------------------- <Directory> -------------------------'
        print 'activity_label: {}'.format(activity_label)

        for id in activities.keys():
            print '\t--------------- <Activity> ---------------'
            print '\tid: {}'.format(id)

            activity = activities[id]
            print '\tactivity_id: {}'.format(activity['activity_id'])
            print '\tobjects: {}'.format(activity['objects'])
            print '\t\tstart_frame: {}'.format(activity['start_frame'])
            print '\t\tend_frame: {}'.format(activity['end_frame'])

            sub_activities = activity['sub_activities']
            for sub_activity in sub_activities:
                print '\t\t----- <Sub-activity> -----'
                print '\t\tsub_activity_id: {}'.format(
                        sub_activity['sub_activity_id'])
                print '\t\tstart_frame: {}'.format(sub_activity['start_frame'])
                print '\t\tend_frame: {}'.format(sub_activity['end_frame'])
                print '\t\taffordances: {}'.format(sub_activity['affordances'])


################################################################################
# Preprocess Functions (Transform)
################################################################################
def preprocess(pos_t, ori_t):
    """
    Preprocess the data representation from raw position and orientation at each
    time step.

    Parameters
    ----------
        pos_t: numpy array
            Position of joints at a time step.
        ori_t: numpy array
            Orientation of joints at a time step.

    Returns
    -------
        datum: numpy array
            Preprocessed data representation of the skeleton.
    """
    # TODO: Start your code here:
    raise Exception('Not implemented yet.')
    datum = np.void
    return datum

def read_and_preprocess(path):
    """
    Preprocess the raw dataset.
    Call read() and preprocess(). 

    Parameters
    ----------
        path: string
            Path of the dataset.

    Returns
    -------
        pos_arr: numpy array
            Positions of joints.
        ori_arr: numpy array
            Orientations of joints.
        data: numpy array
            Preprocessed data representation of skeleton trajectories
        index: list of tuple
            Tuples of (start, end) indices of each sequence. 
    """
    # TODO: Start your code here:
    raise Exception('Not implemented yet.')
    pos_arr = np.void
    data = np.void
    index = []
    return pos_arr, ori_arr, data, index

################################################################################
# Postprocess Functions
################################################################################
def postprocess(datum):
    """
    Postprocess the data representation into raw position and orientation 
    at each time step.

    Parameters
    ----------
        datum: numpy array
            Preprocessed data representation of the skeleton.

    Returns
    -------
        pos_t: numpy array
            Position of joints at a time step.
        ori_t: numpy array
            Orientation of joints at a time step.
    """
    # TODO: Start your code here:
    raise Exception('Not implemented yet.')
    pos_t = np.void
    ori_t = np.void
    return pos_t, ori_t

################################################################################
# 
################################################################################
if __name__ == '__main__':
    # test and print skeleton
    print 'joints: {}'.format(skel['joints'])

    joints = skel['joints']
    tree = skel['tree']
    for j in range(len(tree)):
        print '# {}'.format(joints[j])
        if tree[j]['parent'] == None:
            print '\t<parent>: {}'.format('None')
        else:
            print '\t<parent>: {}'.format(joints[tree[j]['parent']])
        lst = []
        for i in tree[j]['children']:
            lst.append(joints[i])
        print '\t<children>: {}'.format(lst)

    print 'connection: {}'.format(skel['connection'])
    print 'Caution: connection list needs the children apprear in ascending order.'

