import os
import numpy

################################################################################
# Skeleton Parameters
################################################################################
# Index of Joints (0-based)
# TODO: Start your code here:
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

            'root': 2   # root of the skeleton connection tree
            }

# Connection of the Skeleton
# TODO: Start your code here:
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


################################################################################
# IO & Process Parameters
################################################################################
# TODO: Start your code here:
# Add any necessary parameters for loading and processing the dataset. 

################################################################################
# Preprocess Functions
################################################################################
def read(path):
    """
    Read the raw dataset.

    Parameters
    ----------
        path: string
            Path of the dataset.

    Returns
    -------
        All necessary data from the raw dataset. 
        You'd better put all the position in a numpy array, all the orientation
        in a numpy array. And you should create a iterable data structure 
        containing the starting frame and ending frame of each sequence, and all
        other necessary information. 
    """
    # TODO: Start your code here:
    raise Exception('Not implemented yet.')
    return 

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
    datum = numpy.void
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
    pos = numpy.void
    data = numpy.void
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
    pos_t = numpy.void
    ori_t = numpy.void
    return pos_t, ori_t
