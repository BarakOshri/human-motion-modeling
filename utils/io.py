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
    len_seq = sum(1 for line in open(filename))

    ori = numpy.zeros((len_seq, 9*11))
    pos = numpy.zeros((len_seq, 3*15))
    oriconf = numpy.zeros((len_seq, 11))
    posconf = numpy.zeros((len_seq, 15))

    row = 0
    for line in open(filename):
        words = line.split(',')

        if len(words) != 1 + 9*11+3*15+11+15 + 1:
            break

        vals = [float(words[i]) for i in range(1 + 9*11+3*15+11+15)]

        id = float(words[0])
        ori[row, :] = numpy.array([vals[idx] for idx in idx_ori])
        pos[row, :] = numpy.array([vals[idx] for idx in idx_pos])
        oriconf[row, :] = numpy.array([vals[idx] for idx in idx_oriconf])
        posconf[row, :] = numpy.array([vals[idx] for idx in idx_posconf])

        row += 1

    return ori, oriconf, pos, posconf
            

