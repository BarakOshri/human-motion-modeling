import numpy as np
from mlabwrap import mlab
import pickle
#from motion import *
from python_exp2xyz import exp2xyz
import theano
import scipy.io
import pdb

motion_path = "/afs/cs.stanford.edu/u/barak/Workspace/human-motion-modeling/crbm_matlab/Motion/"
mlab.path(mlab.path(), motion_path)

def preprocess(n1, activities_file):
    Motion = mlab.preprocess(n1, activities_file)
    batchdata, seqlen, data_mean, data_std, offsets = _preprocess2_matlab(Motion)

    #shared_x = theano.shared(np.asarray(batchdata, dtype=theano.config.floatX))

    return batchdata, seqlen, data_mean, data_std, offsets

def postprocess(visible, data_std, data_mean, offsets):
	newdata = mlab.postprocess(visible, data_std, data_mean, offsets)

	return newdata

def get_joi(x_t, joi, data_mean, data_std, offsets):
    joi_indices = reduce(lambda x, y: x+y, map(lambda x: range(6*x, 6*(x+1)), joi))

    return postprocess(x_t, data_std.reshape((1, -1)), data_mean.reshape((1, -1)), offsets)[:, joi_indices]

def final_frame(z_t, final_frame_lookahead):

    future = final_frame_lookahead
    num_futures = z_t.shape[0] / final_frame_lookahead

    future_idx = reduce(lambda x, y: x + y, [[(i+1)*future for _ in range(future)] for i in range(num_futures)])
    future_idx += [z_t.shape[0] - 1 for _ in range(z_t.shape[0] % future)]
    future_idx = np.array(future_idx)

    return z_t[future_idx]



# Converts a 2d array of exponential coordinates into the frames' xyz coordinates
# WARNING: EGREGIOUSLY SLOW. Consider using python_exp2xyz for faster runtime,
# though it may be innacurate.
def matlab_exp2xyz(channels):
	xyz = mlab.many_exp2xyz(channels)

	return xyz

# Converts a 2d array of exponential coordinates into the frames' xyz coordinates
def python_exp2xyz(channels):
	return exp2xyz(channels)

def _preprocess2_matlab(Motion):
    n_seq = int(mlab.n_seq(Motion).reshape(1))

    # assume data is MIT format for now
    indx = np.r_[
        np.arange(0,6),
        np.arange(6,9),
        13,
        np.arange(18,21),
        25,
        np.arange(30,33),
        37,
        np.arange(42,45),
        49,
        np.arange(54,57),
        np.arange(60,63),
        np.arange(66,69),
        np.arange(72,75),
        np.arange(78,81),
        np.arange(84,87),
        np.arange(90,93),
        np.arange(96,99),
        np.arange(102,105)]

    row1 = mlab.row1(Motion).reshape((-1))

    offsets =   np.r_[
        row1[None,9:12],
        row1[None,15:18],
        row1[None,21:24],
        row1[None,27:30],
        row1[None,33:36],
        row1[None,39:42],
        row1[None,45:48],
        row1[None,51:54],
        row1[None,57:60],
        row1[None,63:66],
        row1[None,69:72],
        row1[None,75:78],
        row1[None,81:84],
        row1[None,87:90],
        row1[None,93:96],
        row1[None,99:102],
        row1[None,105:108]]

    sequences = [mlab.sequence(Motion, i+1) for i in range(n_seq)]

    # collapse sequences
    batchdata = np.concatenate([m[:, indx] for m in sequences], axis=0)

    data_mean = batchdata.mean(axis=0)
    data_std = batchdata.std(axis=0)

    batchdata = (batchdata - data_mean) / data_std

    # get sequence lengths
    seqlen = [s.shape[0] for s in sequences]

    return batchdata, seqlen, data_mean, data_std, offsets

def _preprocess2_python(Motion):
    n_seq = Motion.shape[1]

    # assume data is MIT format for now
    indx = np.r_[
        np.arange(0,6),
        np.arange(6,9),
        13,
        np.arange(18,21),
        25,
        np.arange(30,33),
        37,
        np.arange(42,45),
        49,
        np.arange(54,57),
        np.arange(60,63),
        np.arange(66,69),
        np.arange(72,75),
        np.arange(78,81),
        np.arange(84,87),
        np.arange(90,93),
        np.arange(96,99),
        np.arange(102,105)]

    row1 = Motion[0,0][0]

    offsets =   np.r_[
        row1[None,9:12],
        row1[None,15:18],
        row1[None,21:24],
        row1[None,27:30],
        row1[None,33:36],
        row1[None,39:42],
        row1[None,45:48],
        row1[None,51:54],
        row1[None,57:60],
        row1[None,63:66],
        row1[None,69:72],
        row1[None,75:78],
        row1[None,81:84],
        row1[None,87:90],
        row1[None,93:96],
        row1[None,99:102],
        row1[None,105:108]]

    # collapse sequences
    batchdata = np.concatenate([m[:, indx] for m in Motion.flat], axis=0)

    data_mean = batchdata.mean(axis=0)
    data_std = batchdata.std(axis=0)

    batchdata = (batchdata - data_mean) / data_std

    # get sequence lengths
    seqlen = [s.shape[0] for s in Motion.flat]

    return batchdata, seqlen, data_mean, data_std, offsets

def load_data(mat_file, n1=250):
    mat_dict = scipy.io.loadmat(mat_file)
    Motion = mat_dict['Motion']

    batchdata, seqlen, data_mean, data_std, offsets = _preprocess2_python(Motion)

    # put data into shared memory
    shared_x = theano.shared(np.asarray(batchdata, dtype=theano.config.floatX))

    return shared_x, seqlen, data_mean, data_std, offsets

def build_plotting_skeleton():
    skel = {}
    skel['joints'] = ['pelvis', 'lfemur', 'ltibia', 'lfoot', 'ltoes', 'rfemur', 'rtibia',
                        'rfoot', 'rtoes', 'thorax', 'lclavicle', 'lhumerus', 'lradius', 'lhand',
                        'rclavicle', 'rhumerus', 'rradius', 'rhand']
    skel['connection'] = [
                            (0, 1), 
                            (0, 5), 
                            (0, 9), 
                            (1, 2), 
                            (2, 3), 
                            (3, 4), 
                            (5, 6), 
                            (6, 7), 
                            (7, 8), 
                            (9, 10), 
                            (9, 14), 
                            (10, 11), 
                            (11, 12), 
                            (12, 13),
                            (14, 15),
                            (15, 16),
                            (16, 17)
                        ]
    skel['tree'] = []
    skel['tree'].append({'children': [1, 5, 9], 'parent': None})
    skel['tree'].append({'children': [2], 'parent': 0})
    skel['tree'].append({'children': [3], 'parent': 1})
    skel['tree'].append({'children': [4], 'parent': 2})
    skel['tree'].append({'children': [], 'parent': 3})
    skel['tree'].append({'children': [6], 'parent': 0})
    skel['tree'].append({'children': [7], 'parent': 5})
    skel['tree'].append({'children': [8], 'parent': 6})
    skel['tree'].append({'children': [], 'parent': 7})
    skel['tree'].append({'children': [10, 14], 'parent': 0})
    skel['tree'].append({'children': [11], 'parent': 9})
    skel['tree'].append({'children': [12], 'parent': 10})
    skel['tree'].append({'children': [13], 'parent': 11})
    skel['tree'].append({'children': [], 'parent': 12})
    skel['tree'].append({'children': [15], 'parent': 9})
    skel['tree'].append({'children': [16], 'parent': 14})
    skel['tree'].append({'children': [17], 'parent': 15})
    skel['tree'].append({'children': [], 'parent': 16})

    return skel


joint_idx = {
            'pelvis': 0,
                'lfemur': 1,
                    'ltibia': 2,
                        'lfoot': 3,
                            'ltoes': 4,
            'rfemur': 5,
                'rtibia': 6,
                    'rfoot': 7,
                        'rtoes': 8,
            'thorax': 9,
                'lclavicle': 10,
                    'lhumerus': 11,
                        'lradius': 12,
                            'lhand': 13,
            'rclavicle': 14,
                'rhumerus': 15,
                    'rradius': 16,
                        'rhand': 17
            }

class MitSkeleton():
    """Skeleton join structure of MIT dataset"""

    def __init__(self):
        self.numNodes = 18
        self.tree = [{} for i in range(self.numNodes)]
        self.type = 'mit'

        self.buildChildren()
        self.buildParents()
        self.buildDataIndices()

    def buildChildren(self):
        self.tree[0]['children'] = [1, 5, 9]
        self.tree[1]['children'] = [2]
        self.tree[2]['children'] = [3]
        self.tree[3]['children'] = [4]
        self.tree[4]['children'] = []
        self.tree[5]['children'] = [6]
        self.tree[6]['children'] = [7]
        self.tree[7]['children'] = [8]
        self.tree[8]['children'] = []
        self.tree[9]['children'] = [10, 14]
        self.tree[10]['children'] = [11]
        self.tree[11]['children'] = [12]
        self.tree[12]['children'] = [13]
        self.tree[13]['children'] = []
        self.tree[14]['children'] = [15]
        self.tree[15]['children'] = [16]
        self.tree[16]['children'] = [17]
        self.tree[17]['children'] = []

    def buildParents(self):
        self.tree[0]['parent'] = None
        self.tree[1]['parent'] = 0
        self.tree[2]['parent'] = 1
        self.tree[3]['parent'] = 2
        self.tree[4]['parent'] = 3
        self.tree[5]['parent'] = 0
        self.tree[6]['parent'] = 5
        self.tree[7]['parent'] = 6
        self.tree[8]['parent'] = 7
        self.tree[9]['parent'] = 0
        self.tree[10]['parent'] = 9
        self.tree[11]['parent'] = 10
        self.tree[12]['parent'] = 11
        self.tree[13]['parent'] = 12
        self.tree[14]['parent'] = 9
        self.tree[15]['parent'] = 14
        self.tree[16]['parent'] = 15
        self.tree[17]['parent'] = 16

    def buildDataIndices(self):
        for i in range(self.numNodes):
            self.tree[i]['or'] = np.arange(i*6, i*6+3)
            self.tree[i]['offset'] = np.arange(i*6+3, (i+1)*6)

if __name__ == "__main__":
	shared_x, seqlen, data_mean, data_std, offsets = preprocess(10, 'data/activities_file.txt')
	print shared_x.shape
	newdata = postprocess(shared_x, data_std, data_mean, offsets)
	xyz = python_exp2xyz(newdata)
