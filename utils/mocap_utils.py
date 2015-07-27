import numpy as np

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
    invR = np.linalg.inv(R)

    for i in range(pos.shape[0]):
        for j in range(pos.shape[1]/3):
            pos_new[i, j*3:(j+1)*3] = np.dot(invR, pos[i, j*3:(j+1)*3] - t)

        for j in range(ori.shape[1]/9):
            ori_new[i, j*9:(j+1)*9] \
                = (invR * ori[i, j*9:(j+1)*9].reshape(3, 3)).reshape(1, 9)

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
