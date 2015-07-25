import numpy

def joint_pos(pos_t, idx):
    """
    Return the position of the idx'th joint. 
    """
    return numpy.array([pos_t[idx*3+0], pos_t[idx*3+1], pos_t[idx*3+2]])

def joint_ori(ori_t, idx):
    """
    Return the orientation matrix of the idx'th joint. 
    """
    return numpy.array([[ori_t[idx*3+0], ori_t[idx*3+1], ori_t[idx*3+2]], 
                        [ori_t[idx*3+3], ori_t[idx*3+4], ori_t[idx*3+5]], 
                        [ori_t[idx*3+6], ori_t[idx*3+7], ori_t[idx*3+8]]])

def change_space(pos, ori, t = numpy.zeros(3), R = numpy.eye(3)):
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
    pos_new = numpy.zeros(pos.shape)
    ori_new = numpy.zeros(ori.shape)
    invR = numpy.linalg.inv(R)

    for i in range(pos.shape[0]):
        for j in range(pos.shape[1]/3):
            pos_new[i, j*3:(j+1)*3] = numpy.dot(invR, pos[i, j*3:(j+1)*3] - t)

        for j in range(ori.shape[1]/9):
            ori_new[i, j*9:(j+1)*9] \
                = (invR * ori[i, j*9:(j+1)*9].reshape(3, 3)).reshape(1, 9)

    return pos_new, ori_new

def get_offset_t(pos_t, ori_t, connection):
    """
    Return the offset arrays at the step t.
    """
    offset_t = numpy.zeros(pos_t.shape[0]-3)
    i = 0; # store the offset according to the order of connection list
    for (parent, child) in connection:
        t = joint_pos(pos_t, parent)
        R = joint_ori(ori_t, parent)
        offset_t[None, 3*i:3*(i+1)] \
            = numpy.dot(joint_pos(pos_t, child) - t, R.T)
        i += 1
    return offset_t

def get_mean_offset(pos, ori, connection):
    """
    Return the mean offset arrays computed from position and orientation arrays.
    """
    offset = numpy.zeros((pos.shape[0], pos.shape[1]-3))
    for t in range(pos.shape[0]):
        offset[t, :] = get_offset_t(pos[t, :], ori[t, :], connection)
    return numpy.mean(offset, axis=0)
