import numpy as np
from space import *

def joint_pos(pos_t, idx):
    """
    Return the position of the idx'th joint. 
    """
    return np.array([pos_t[idx*3+0], pos_t[idx*3+1], pos_t[idx*3+2]])

def joint_ori(ori_t, idx):
    """
    Return the orientation matrix of the idx'th joint. 
    """
    return np.array([[ori_t[idx*3+0], ori_t[idx*3+1], ori_t[idx*3+2]], 
                        [ori_t[idx*3+3], ori_t[idx*3+4], ori_t[idx*3+5]], 
                        [ori_t[idx*3+6], ori_t[idx*3+7], ori_t[idx*3+8]]])

def change_space(pos, ori, t = np.zeros(3), R = np.eye(3)):
    """
    Chang position and orientation arrays into a new coordinate system.

    Parameters
    ----------
    pos: numpy array
    Position array in the current space.
    ori:
    Orientation array in the current space.
    t:
    Translation vector of the origin of the new space.
    R:
    Rotation matrix of the new space.
    """
    pos_new = np.zeros(pos.shape)
    ori_new = np.zeros(ori.shape)

    for i in range(pos.shape[0]):
        for j in range(pos.shape[1]/3):
            pos_new[i, j*3:(j+1)*3] = np.dot(pos[i, j*3:(j+1)*3] - t, R.T)

        for j in range(ori.shape[1]/9):
            ori_new[i, j*9:(j+1)*9] \
                = np.dot(np.dot(R, ori[i, j*9:(j+1)*9].reshape(3, 3)), R.T).reshape(1, 9)

    return pos_new, ori_new

def get_offset_t(pos_t, ori_t, connection):
    """
    Return the offset arrays at the step t.
    """
    offset_t = np.zeros(pos_t.shape[0]-3)
    i = 0; # store the offset according to the order of connection list
    for (parent, child) in connection:
        t = joint_pos(pos_t, parent)
        R = joint_ori(ori_t, parent)
        offset_t[None, 3*i:3*(i+1)] \
            = np.dot(joint_pos(pos_t, child) - t, R.T)
        i += 1
    return offset_t

def get_mean_offset(pos, ori, connection):
    """
    Return the mean offset arrays computed from position and orientation arrays.
    """
    offset = np.zeros((pos.shape[0], pos.shape[1]-3))
    for t in range(pos.shape[0]):
        offset[t, :] = get_offset_t(pos[t, :], ori[t, :], connection)
    return np.mean(offset, axis=0)

def ori2pos(joint_idx, connection, pos_root, ori, offset):
    """
    Recover position of joints from the orientation.
    """
    pos = np.zeros((pos_root.shape[0], 3*len(joint_idx)))

    # root joint's position and orientation
    root = joint_idx['torso']
    pos[:, 3*root:3*(root+1)] = pos_root

    # other joint's position and orientation
    for t in range(pos.shape[0]):
        i = 0
        for (parent, child) in connection:
            R = joint_ori(ori[t, :], parent)
            pos_parent = joint_pos(pos[t, :], parent)
            offset_i = offset[3*i:3*(i+1)]
            pos[t, 3*child:3*(child+1)] \
                = pos_parent + np.dot(offset_i, R)
            i += 1
        return pos

def preprocess_joi(joint_idx, list_joi, pos, ori):
    """
    Preprocess to get data representations of joints of interest. 
    """
    torso = joint_idx['torso']
    data = np.zeros((pos.shape[0], 6+3*len(list_joi))).astype('float32')
    for t in range(pos.shape[0]):
        pos_torso = joint_pos(pos[t, :], torso)
        ori_torso = joint_ori(ori[t, :], torso)
        data[t, 0:3] = pos_torso
        data[t, 3:6] = rmat_to_r3(ori_torso)
        i = 0
        for j in list_joi: 
            pos_j = joint_pos(pos[t, :], joint_idx[j])
            # Convert to body-centered
            data[t, 6+3*i:9+3*i] = \
                pos_transform(pos_j, pos_torso, ori_torso)
            i += 1
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data = (data - mean) / std

    pos_joi = np.concatenate([pos[:, 3*joint_idx[j]:3*(joint_idx[j]+1)] 
                                for j in ['torso']+list_joi], 
                                axis=1)
    return data, mean, std, pos_joi

def postprocess_joi(joint_idx, data, mean, std):
    """
    Postprocess to get the positions of joints of interest. 
    """
    data = data * std + mean

    pos_joi = np.zeros((data.shape[0], data.shape[1]-3), ).astype('float32')
    pos_joi[:, 0:3] = data[:, 0:3]
    
    for t in range(data.shape[0]):
        pos_torso = data[t, 0:3]
        ori_torso = r3_to_rmat(data[t, 3:6])
        for i in range((data.shape[1]-3)/3-1):
            pos_joi[t, 3+3*i:6+3*i] = \
                pos_inv_transform(data[t, 6+3*i:9+3*i], pos_torso, ori_torso)

    return pos_joi

def preprocess_relpos(joint_idx, connection, pos, ori):
    """
    Preprocess the raw position and orientation into relative positins. 
    """
    torso = joint_idx['torso']
    data = np.zeros((pos.shape[0], 3+pos.shape[1])).astype('float32')
    for t in range(pos.shape[0]):
        pos_torso = joint_pos(pos[t, :], torso)
        ori_torso = joint_ori(ori[t, :], torso)
        data[t, 0:3] = pos_torso
        data[t, 3:6] = rmat_to_r3(ori_torso)

        i = len(connection)-1
        for (p, c) in reversed(connection):
            pos_p = joint_pos(pos[t, :], p)
            pos_c = joint_pos(pos[t, :], c)
            # Convert to body-centered relative value
            data[t, 6+3*i:9+3*i] = \
                pos_transform(pos_c, pos_p, ori_torso) # TODO: better way?
            i -= 1

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data = (data - mean) / std
    return data, mean, std

def postprocess_relpos(joint_idx, connection, data, mean, std):
    """
    Postprocessing relative positions into world positions. 
    """
    torso = joint_idx['torso']

    data = data * std + mean
    pos = np.zeros((data.shape[0], data.shape[1]-3), ).astype('float32')
    pos[:, 3*torso:3*(torso+1)] = data[:, 0:3]
    for t in range(data.shape[0]):
        pos_torso = data[t, 0:3]
        ori_torso = r3_to_rmat(data[t, 3:6])

        i = 0
        for (p, c) in connection:
            pos_p = joint_pos(pos[t, :], p)
            offset = data[t, 6+3*i:9+3*i]
            # Convert to body-centered relative value
            pos_c = pos_inv_transform(offset, pos_p, ori_torso)
            pos[t, 3*c:3*(c+1)] = pos_c
            i += 1

    return pos
