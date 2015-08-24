import os
import numpy as np
from collections import OrderedDict 
from spacepy import pycdf

################################################################################
# Skeleton Parameters
################################################################################
skel = {}

# Names of Joints
joints = [
            ]
skel['joints'] = joints

tree = []
# torso
tree.append({
            })
# neck

skel['tree'] = tree

connection = [(tree[c]['parent'], c) for c in range(1, len(joints))]
skel['connection'] = connection

################################################################################
# IO & Process Parameters
################################################################################

################################################################################
# Preprocess Functions (Read)
################################################################################
def read_subject(path):
    """
    Read a subject
    """
    # TODO
    return 

def read_pos(path):
    """
    Read the D3 positions file of a subject.
    """
    cdf = pycdf.CDF(path)
    return np.array(cdf)


################################################################################
# 
################################################################################
if __name__ == '__main__':
    path_data = 'data/'
    # pos = read_pos('data/h36m/S1/MyPoseFeatures/D3_Angles/Eating.cdf')
    pos = read_pos('data/h36m/S1/MyPoseFeatures/D3_Angles/Discussion.cdf')
    print pos
    print np.array(pos)


