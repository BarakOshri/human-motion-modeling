import os
import numpy as np

################################################################################
# Skeleton Parameters
################################################################################
# Index of Joints
joint_idx = {
            'head': 0, 
            'neck': 1, 
            'torso': 2, 
            'left_shoulder': 3, 
            'left_elbow': 4, 
            'right_shoulder': 5, 
            'right_elbow': 6, 
            'left_hip': 7, 
            'left_knee': 8, 
            'right_hip': 9, 
            'right_knee': 10, 
            'left_hand': 11, 
            'right_hand': 12, 
            'left_foot': 13, 
            'right_foot': 14,
            }

# Original code of computing index of joints:
# # 1-based index
# joint_idx_1based = {}
# joint_idx_1based['head']            = 1
# joint_idx_1based['neck']            = 2
# joint_idx_1based['torso']           = 3
# joint_idx_1based['left_shoulder']   = 4
# joint_idx_1based['left_elbow']      = 5
# joint_idx_1based['right_shoulder']  = 6
# joint_idx_1based['right_elbow']     = 7
# joint_idx_1based['left_hip']        = 8
# joint_idx_1based['left_knee']       = 9
# joint_idx_1based['right_hip']       = 10
# joint_idx_1based['right_knee']      = 11
# joint_idx_1based['left_hand']       = 12
# joint_idx_1based['right_hand']      = 13
# joint_idx_1based['left_foot']       = 14
# joint_idx_1based['right_foot']      = 15
# 
# # 0-based index
# joint_idx = {}
# for joint_name in joint_idx_1based.keys():
#     joint_idx[joint_name] = joint_idx_1based[joint_name] - 1


# Connection of the Skeleton
connection = [
            # breath 1 
            (2, 1), 
            (2, 3), 
            (2, 5), 
            (2, 7), 
            (2, 9), 
            
            # breath 2
            (1, 0), 
            (3, 4), 
            (5, 6), 
            (7, 8), 
            (9, 10), 

            # breath 3
            (4, 11), 
            (6, 12), 
            (8, 13), 
            (10, 14)
            ]

# Original code of computing connection of the skeleton:
# joint_connect = [\
#   ('torso', 'neck'), ('torso', 'left_shoulder'), ('torso', 'right_shoulder'), 
#     ('torso', 'left_hip'), ('torso', 'right_hip'), 
# 
#     ('neck', 'head'),
#     ('left_shoulder', 'left_elbow'),
#     ('right_shoulder', 'right_elbow'),
#     ('left_hip', 'left_knee'), 
#     ('right_hip', 'right_knee'), 
# 
#     ('left_elbow', 'left_hand'),
#     ('right_elbow', 'right_hand'),
#     ('left_knee', 'left_foot'),
#     ('right_knee', 'right_foot')]
# 
# root = joint_idx['torso']
# 
# # 0-based index
# connect = [(joint_idx[parent], joint_idx[child])
#             for parent, child in joint_connect]


################################################################################
# IO & Process Parameters
################################################################################
# Add any necessary parameters for loading and processing the dataset. 
idx_ori = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 29,
            30, 31, 32, 33, 34, 35, 36, 37, 43, 44, 45, 46, 47, 48, 49, 50, 51,
            57, 58, 59, 60, 61, 62, 63, 64, 65, 71, 72, 73, 74, 75, 76, 77, 78,
            79, 85, 86, 87, 88, 89, 90, 91, 92, 93, 99, 100, 101, 102, 103, 104,
            105, 106, 107, 113, 114, 115, 116, 117, 118, 119, 120, 121, 127, 
            128, 129, 130, 131, 132, 133, 134, 135, 141, 142, 143, 144, 145, 
            146, 147, 148, 149]
idx_pos = [11, 12, 13, 25, 26, 27, 39, 40, 41, 53, 54, 55, 67, 68, 69, 81, 82, 
            83, 95, 96, 97, 109, 110, 111, 123, 124, 125, 137, 138, 139, 151, 
            152, 153, 155, 156, 157, 159, 160, 161, 163, 164, 165, 167, 168, 
            169] 
idx_oriconf = [10, 24, 38, 52, 66, 80, 94, 108, 122, 136, 150]
idx_posconf = [14, 28, 42, 56, 70, 84, 98, 112, 126, 140, 154, 158, 162, 166,
                170]

# Original code for computing parameters for loading data:
# idx_ori = []#[None] * (9*11)
# idx_pos = []#[None] * (3*15)
# idx_oriconf =  []#[None] * (11)
# idx_posconf =  []#[None] * (15)
# 
# for i in range(11):
#     start = 1 + i*(9+1+3+1)
#     for j in range(9):
#         idx_ori.append(start + j)
#     idx_oriconf.append(start + 9)
#     for j in range(3):
#         idx_pos.append(start + 9 + 1 + j)
#     idx_posconf.append(start + 9 + 1 + 3)
# 
# for i in range(11, 15):
#     start = 1 + 11*(9+1+3+1) + (i-11)*(3+1)
#     for j in range(3):
#         idx_pos.append(start + j)
#     idx_posconf.append(start + 3)
# 
# print idx_ori
# print idx_pos
# print idx_oriconf
# print idx_posconf



################################################################################
# Preprocess Functions (Read)
################################################################################
def read(path):
    """
    Read a subject
    """
    pos, posconf, ori, oriconf, subject = merge(_read_subject(path))
    return pos, posconf, ori, oriconf, subject


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
    len_seq = sum(1 for line in open(filename))-1

    ori = np.zeros((len_seq, 9*11), dtype = np.float32)
    pos = np.zeros((len_seq, 3*15), dtype = np.float32)
    oriconf = np.zeros((len_seq, 11), dtype = np.float32)
    posconf = np.zeros((len_seq, 15), dtype = np.float32)

    row = 0
    for line in open(filename):
        words = line.split(',')

        if len(words) != 1 + 9*11+3*15+11+15 + 1:
            break

        vals = [float(words[i]) for i in range(1 + 9*11+3*15+11+15)]

        id = float(words[0])
        ori[row, :] = np.array([vals[idx] for idx in idx_ori])
        pos[row, :] = np.array([vals[idx] for idx in idx_pos])
        oriconf[row, :] = np.array([vals[idx] for idx in idx_oriconf])
        posconf[row, :] = np.array([vals[idx] for idx in idx_posconf])

        row += 1

    return ori, oriconf, pos, posconf


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
        ori, oriconf, pos, posconf = _read_skeleton_data(path_subact)
        
        activity['ori'] = ori
        activity['oriconf'] = oriconf
        activity['pos'] = pos
        activity['posconf'] = posconf

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
        sub_activity['end_frame'] = end_frame - 1       # 1-based to 0-based
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
    Merge the pos's and ori's into the same numpy array. 
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
            act_len = activity['ori'].shape[0]
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


    pos = np.concatenate(\
            [activity['pos']
                for activity_label, directory in subject.iteritems()
                for id, activity in directory['activities'].iteritems()
            ],
            axis=0)

    ori = np.concatenate(\
            [activity['ori']
                for activity_label, directory in subject.iteritems()
                for id, activity in directory['activities'].iteritems()
            ],
            axis=0)

    posconf = np.concatenate(\
            [activity['posconf']
                for activity_label, directory in subject.iteritems()
                for id, activity in directory['activities'].iteritems()
            ],
            axis=0)

    oriconf = np.concatenate(\
            [activity['oriconf']
                for activity_label, directory in subject.iteritems()
                for id, activity in directory['activities'].iteritems()
            ],
            axis=0)

    return pos, posconf, ori, oriconf, subject_new


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
def preprocess(joint_idx, connection, pos_t, ori_t):
    """
    Preprocess the data representation from raw position and orientation at each
    time step.

    Parameters
    ----------
        joint_idx: dict
            Index of joints. 
        connection: list of tuples
            Connection of joints in the skeleton. 
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
        pos: numpy array
            Position of joints.
        ori: numpy array
            Orientation of joints.
        data: numpy array
            Preprocessed data representation of skeleton trajectories
        index: list of tuple
            Tuples of (start, end) indices of each sequence. 
    """
    # TODO: Start your code here:
    raise Exception('Not implemented yet.')
    pos = np.void
    data = np.void
    index = []
    return pos, ori, data, index

################################################################################
# Postprocess Functions
################################################################################
def postprocess(joint_idx, connection, datum):
    """
    Postprocess the data representation into raw position and orientation 
    at each time step.

    Parameters
    ----------
        joint_idx: dict
            Index of joints. 
        connection: list of tuples
            Connection of joints in the skeleton. 
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
