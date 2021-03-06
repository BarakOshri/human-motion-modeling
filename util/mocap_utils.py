import numpy as np
from space import *

def pos_ind(i):
    return np.array(range(i*3,(i+1)*3))

def ori_ind(i):
    return np.array(range(i*9,(i+1)*9))

def joint_pos(pos_t, ind):
    """
    Return the position of the ind'th joint. 
    """
    return np.array([pos_t[ind*3+0], pos_t[ind*3+1], pos_t[ind*3+2]])

def joint_ori(ori_t, ind):
    """
    Return the orientation matrix of the ind'th joint. 
    """
    return ori_t[ind*9:(ind+1)*9].reshape(3, 3)

def change_space(pos_arr, ori_arr, trans = np.zeros(3), R = np.eye(3)):
    """
    Chang position and orientation arrays into a new coordinate system.

    Parameters
    ----------
    pos_arr: numpy array
        Position array in the current space.
    ori_arr:
        Orientation array in the current space.
    t:
        Translation vector of the origin of the new space.
    R:
        Rotation matrix of the new space.
    """
    pos_arr_new = np.empty(pos_arr.shape)
    ori_arr_new = np.empty(ori_arr.shape)

    for t in range(pos_arr.shape[0]):
        for j in range(pos_arr.shape[1]/3):
            v = pos_arr[t, pos_ind(j)]
            pos_arr_new[t, pos_ind(j)] = np.dot(v - trans, R.T)

        for j in range(ori_arr.shape[1]/9):
            o = ori_arr[t, ori_ind(j)].reshape(3, 3)
            ori_arr_new[t, ori_ind(j)] = np.dot(np.dot(R, o), R.T).reshape(1, 9)

    return pos_arr_new, ori_arr_new

def get_offset_t(skel, pos_t, ori_t):
    """
    Return the offset arrays at the step t.
    """
    connection = skel['connection']

    offset_t = np.empty(pos_t.shape[0]-3)
    i = 0; # store the offset according to the order of connection list
    for (parent, child) in connection:
        t = joint_pos(pos_t, parent)
        R = joint_ori(ori_t, parent)
        offset_t[None, i*3:(i+1)*3] \
            = np.dot(joint_pos(pos_t, child) - t, R.T)
        i += 1
    return offset_t

def get_offset_arr(skel, pos_arr, ori_arr):
    """
    Return the mean offset arrays computed from position and orientation arrays.
    """
    offset_arr = np.concatenate(
                [[get_offset_t(skel, pos_arr[t, :], ori_arr[t, :])]\
                for t in range(pos_arr.shape[0])], axis=0)
    return offset_arr

def assemble(skel, pos_torso_t, ori_t, offset_t):
    """
    Assemble torso positions and offsets into the world positions. 
    """
    connection = skel['connection']
    joints = skel['joints']

    pos_t = np.empty(len(joints)*3)
    pos_t[0:3] = pos_torso_t

    i = 0; # store the offset according to the order of connection list
    for (parent, child) in connection:
        t = offset_t[i*3:(i+1)*3]
        R = joint_ori(ori_t, parent)
        pos_t[child*3:(child+1)*3] = pos_t[parent*3:(parent+1)*3] + np.dot(t, R)
        i += 1

    return pos_t

def assemble_all(skel, pos_torso_arr, ori_arr, offset_arr):
    pos_arr = np.concatenate(
                            [[assemble(skel, pos_torso_arr[t, :], 
                            ori_arr[t, :], offset_arr[t, :])]\
                            for t in range(ori_arr.shape[0])], axis=0)
    return pos_arr

def ori2pos(skel, pos_torso_arr, ori_arr, offset_t):
    """
    Recover position of joints from the orientation.
    """
    connection = skel['connection']
    joints = skel['joints']

    pos_arr = np.zeros((pos_torso_arr.shape[0], 3*len(joints)))

    # torso joint's position and orientation
    pos_arr[:, 0:3] = pos_torso_arr

    # other joint's position and orientation
    for t in range(pos_arr.shape[0]):
        i = 0
        for (parent, child) in connection:
            R = joint_ori(ori_arr[t, :], parent)
            pos_parent = joint_pos(pos_arr[t, :], parent)
            offset_i = offset_t[3*i:3*(i+1)]
            pos_arr[t, 3*child:3*(child+1)] \
                = pos_parent + np.dot(offset_i, R)
            i += 1
        return pos_arr

def get_loss_direct(data, index):
    """
    Directly using the current frame as the prediction, compute the loss. 
    """
    loss_direct = 0
    for i in range(index.shape[0]):
        start = index[i, 0]
        end = index[i, 1]
        din = data[start:end-1, :]
        dout = data[start+1:end, :]
        loss_direct += np.sum((dout - din)**2) / data.shape[1]
    loss_direct = np.sqrt(loss_direct / index.shape[0])
    return loss_direct

def preprocess_joi(list_joi, pos_arr, ori_torso_arr):
    """
    Preprocess to get data representations of joints of interest. 
    """
    data = np.zeros((pos_arr.shape[0], 6+3*len(list_joi))).astype('float32')
    pos_torso_arr = pos_arr[:, pos_ind(0)]

    for t in range(pos_arr.shape[0]):
        rmat_torso = ori_torso_arr[t, :].reshape((3, 3))
        data[t, 0:3] = pos_torso_arr[t, :]
        data[t, 3:6] = rmat_to_r3(rmat_torso)

        i = 0
        for j in list_joi: 
            pos_j = pos_arr[t, pos_ind(j)] 
            # Convert to body-centered
            data[t, 6+3*i:9+3*i] = \
                pos_transform(pos_j, pos_torso_arr[t, :], rmat_torso)
            i += 1

    pos_joi_arr = np.concatenate([pos_arr[:, pos_ind(j)] 
                                for j in [0]+list_joi], 
                                axis=1)
    return data, pos_joi_arr

def postprocess_joi(data):
    """
    Postprocess to get the positions of joints of interest. 
    """
    pos_joi_arr = np.zeros((data.shape[0], data.shape[1]-3), ).astype('float32')
    pos_joi_arr[:, pos_ind(0)] = data[:, 0:3]
    
    for t in range(data.shape[0]):
        ori_torso = r3_to_rmat(data[t, 3:6])
        for i in range((data.shape[1]-3)/3-1):
            pos_joi_arr[t, pos_ind(1+i)] = \
                pos_inv_transform(data[t, 6+3*i:9+3*i], data[t, 0:3], ori_torso)

    return pos_joi_arr

def preprocess_bcpos(skel, pos_arr, ori_torso_arr):
    """
    Preprocess the raw position and orientation into body-cente positins. 
    """
    connection = skel['connection']

    bcpos_arr = np.zeros((pos_arr.shape[0], 3+pos_arr.shape[1])).astype('float32')
    for t in range(pos_arr.shape[0]):
        pos_torso_t = pos_arr[t, pos_ind(0)]
        rmat_torso = ori_torso_arr[t, :].reshape((3, 3))
        bcpos_arr[t, 0:3] = pos_torso_t
        bcpos_arr[t, 3:6] = rmat_to_r3(rmat_torso)

        i = len(connection)-1
        for (p, c) in reversed(connection):
            pos_p = pos_arr[t, pos_ind(p)]
            pos_c = pos_arr[t, pos_ind(c)]
            # Convert to body-centered relative value
            bcpos_arr[t, 6+3*i:9+3*i] = pos_transform(pos_c, pos_p, rmat_torso) 
            # TODO: better way?
            i -= 1

    return bcpos_arr

def postprocess_bcpos(skel, bcpos_arr):
    """
    Postprocess relative positions into world positions. 
    """
    connection = skel['connection']

    pos_arr = np.zeros((bcpos_arr.shape[0], bcpos_arr.shape[1]-3), ).astype('float32')
    pos_arr[:, pos_ind(0)] = bcpos_arr[:, 0:3]
    for t in range(bcpos_arr.shape[0]):
        pos_torso_arr = bcpos_arr[t, 0:3]
        ori_torso = r3_to_rmat(bcpos_arr[t, 3:6])

        i = 0
        for (p, c) in connection:
            pos_p = joint_pos(pos_arr[t, :], p)
            offset = bcpos_arr[t, 6+3*i:9+3*i]
            # Convert to body-centered relative value
            pos_c = pos_inv_transform(offset, pos_p, ori_torso)
            pos_arr[t, 3*c:3*(c+1)] = pos_c
            i += 1

    return pos_arr

def preprocess_relexpmap_dev(skel, pos_arr, ori_arr):
    """
    Preprocess the pos_arr and ori_arr into relative exponential maps and 
    offset deviations.
    """
    joints = skel['joints']
    connection = skel['connection']

    pos_torso_arr = pos_arr[:, pos_ind(0)]
    expmap_torso_arr = np.concatenate(
                    [[rmat_to_r3(ori_arr[t, ori_ind(0)].reshape((3, 3)))]\
                    for t in range(ori_arr.shape[0])], axis=0)
    
    relpos_arr = np.empty((pos_arr.shape[0], 3*len(connection)))
    relexpmap_arr = np.empty((pos_arr.shape[0], 3*(len(connection)-4)))

    for t in range(pos_arr.shape[0]):
        i = 0
        for (p, c) in connection:
            pos_p = pos_arr[t, pos_ind(p)]
            pos_c = pos_arr[t, pos_ind(c)]
            rmat_p = ori_arr[t, ori_ind(p)].reshape(3, 3)
            pos_c_in_p = np.dot((pos_c - pos_p), rmat_p.T)
            relpos_arr[t, i*3:(i+1)*3] = pos_c_in_p

            if i < len(connection) - 4:
                rmat_c = ori_arr[t, ori_ind(c)].reshape(3, 3)
                relrmat = np.dot(np.dot(rmat_p, rmat_c), rmat_p.T)
                relexpmap = rmat_to_r3(relrmat)
                relexpmap_arr[t, i*3:(i+1)*3] = relexpmap
            i += 1

    offset_arr = get_offset_arr(skel, pos_arr, ori_arr)
    offset_mean = np.mean(offset_arr, axis=0)
    dev_arr = relpos_arr - offset_mean

    data = np.concatenate([pos_torso_arr, expmap_torso_arr, relexpmap_arr, 
                            dev_arr], axis=1)
    return data, offset_mean

def postprocess_relexpmap_dev(skel, data, offset_mean):
    """
    Postprocess the relative exponential maps and 
    offset deviations back into pos_arr and ori_arr.
    """
    joints = skel['joints']
    connection = skel['connection']

    relexpmap_arr = data[:, 6:6+3*(len(connection)-4)]
    dev_arr = data[:, 6+3*(len(connection)-4):]
    
    pos_arr = np.empty((data.shape[0], 3*len(joints))) 
    ori_arr = np.empty((data.shape[0], 9*(len(joints)-4))) 

    pos_arr[:, pos_ind(0)] = data[:, 0:3]
    ori_arr[:, ori_ind(0)] = np.concatenate(
                                [r3_to_rmat(data[t, 3:6]).reshape(1, 9)\
                                for t in range(data.shape[0])], axis=0)
    
    for t in range(pos_arr.shape[0]):
        i = 0 
        for (p, c) in connection:
            pos_p = pos_arr[t, pos_ind(p)]
            rmat_p = ori_arr[t, ori_ind(p)].reshape(3, 3)
            pos_c_in_p = offset_mean[i*3:(i+1)*3] + dev_arr[t, i*3:(i+1)*3]
            pos_c = pos_p + np.dot(pos_c_in_p, rmat_p)
            pos_arr[t, c*3:(c+1)*3] = pos_c

            if c < len(joints) - 4:
                relrmat = r3_to_rmat(relexpmap_arr[t, i*3:(i+1)*3])
                rmat_c = np.dot(rmat_p.T, np.dot(relrmat, rmat_p))
                ori_c = rmat_c.reshape(1, 9)
                ori_arr[t, c*9:(c+1)*9] = ori_c

            i += 1

    return pos_arr, ori_arr

def preprocess_relexpmap_dev_perseq(skel, index, pos_arr, ori_arr, 
                                    has_dev=True):
    """
    """
    joints = skel['joints']
    connection = skel['connection']

    # offset_arr = get_offset_arr(skel, pos_arr, ori_arr)
    # offset_mean = np.mean(offset_arr, axis=0)

    pos_torso_arr = pos_arr[:, pos_ind(0)]
    expmap_torso_arr = np.concatenate(
                    [[rmat_to_r3(ori_arr[t, ori_ind(0)].reshape((3, 3)))]\
                    for t in range(ori_arr.shape[0])], axis=0)
    
    relpos_arr = np.empty((pos_arr.shape[0], 3*len(connection)))
    relexpmap_arr = np.empty((pos_arr.shape[0], 3*(len(connection)-4)))

    for t in range(pos_arr.shape[0]):
        j = 0
        for (p, c) in connection:
            pos_p = pos_arr[t, pos_ind(p)]
            pos_c = pos_arr[t, pos_ind(c)]
            rmat_p = ori_arr[t, ori_ind(p)].reshape(3, 3)
            pos_c_in_p = np.dot((pos_c - pos_p), rmat_p.T)
            relpos_arr[t, j*3:(j+1)*3] = pos_c_in_p

            if j < len(connection) - 4:
                rmat_c = ori_arr[t, ori_ind(c)].reshape(3, 3)
                relrmat = np.dot(np.dot(rmat_p, rmat_c), rmat_p.T)
                relexpmap = rmat_to_r3(relrmat)
                relexpmap_arr[t, j*3:(j+1)*3] = relexpmap
            j += 1

    offset_mean_arr = np.empty((index.shape[0], 3*len(connection)))
    dev_arr = np.empty((pos_arr.shape[0], 3*len(connection)))
    for i in range(index.shape[0]):
        start = index[i, 0]
        end = index[i, 1]
        offset_mean_arr[i, :] = np.mean(
            get_offset_arr(skel, pos_arr[start:end, :], ori_arr[start:end, :]), 
            axis=0)
        dev_arr[start:end, :] = relpos_arr[start:end, :] - offset_mean_arr[i, :]

    if has_dev: 
        data = np.concatenate([pos_torso_arr, expmap_torso_arr, relexpmap_arr, 
                                dev_arr], axis=1)
    else:
        data = np.concatenate([pos_torso_arr, expmap_torso_arr, relexpmap_arr], 
                                axis=1)
    return data, offset_mean_arr

def postprocess_relexpmap_dev_perseq(skel, index, data, offset_mean_arr, 
                                    has_dev=True):
    """
    """
    joints = skel['joints']
    connection = skel['connection']

    relexpmap_arr = data[:, 6:6+3*(len(connection)-4)]

    if has_dev: 
        dev_arr = data[:, 6+3*(len(connection)-4):]
    else:
        dev_arr = np.zeros((data.shape[0], 3*len(connection)))
    
    pos_arr = np.empty((data.shape[0], 3*len(joints))) 
    ori_arr = np.empty((data.shape[0], 9*(len(joints)-4))) 

    pos_arr[:, pos_ind(0)] = data[:, 0:3]
    ori_arr[:, ori_ind(0)] = np.concatenate(
                                [r3_to_rmat(data[t, 3:6]).reshape(1, 9)\
                                for t in range(data.shape[0])], axis=0)
    
    for i in range(index.shape[0]):
        start = index[i, 0]
        end = index[i, 1]
        for t in range(start, end):
            j = 0 
            for (p, c) in connection:
                pos_p = pos_arr[t, pos_ind(p)]
                rmat_p = ori_arr[t, ori_ind(p)].reshape(3, 3)
                pos_c_in_p = offset_mean_arr[i, j*3:(j+1)*3] +\
                             dev_arr[t, j*3:(j+1)*3]
                pos_c = pos_p + np.dot(pos_c_in_p, rmat_p)
                pos_arr[t, c*3:(c+1)*3] = pos_c

                if c < len(joints) - 4:
                    relrmat = r3_to_rmat(relexpmap_arr[t, j*3:(j+1)*3])
                    rmat_c = np.dot(rmat_p.T, np.dot(relrmat, rmat_p))
                    ori_c = rmat_c.reshape(1, 9)
                    ori_arr[t, c*9:(c+1)*9] = ori_c

                j += 1

    return pos_arr, ori_arr

def abs2inc(seq_abs):
    """
    Transform from absolute value into incremental value. 
    """
    seq_inc = np.empty((seq_abs.shape[0]-1, seq_abs.shape[1])).astype('float32')
    for t in range(1, seq_abs.shape[0]):
        seq_inc[t-1, :] = seq_abs[t, :] - seq_abs[t-1, :]
    return seq_inc

def inc2abs(init, seq_inc):
    """ 
    Transform from incremental value into absolute value. 
    """
    seq_abs = np.empty((seq_inc.shape[0]+1, seq_inc.shape[1])).astype('float32')
    seq_abs[0, :] = init
    for t in range(1, seq_abs.shape[0]):
        seq_abs[t, :] = seq_abs[t-1, :] + seq_inc[t-1, :]
    return seq_abs

def abs2inc_forall(data_abs, index_abs):
    data_inc = np.concatenate(
                [abs2inc(data_abs[index_abs[i, 0]:index_abs[i, 1], :])\
                for i in range(index_abs.shape[0])],
                axis=0)
    inits = np.concatenate(
                            [data_abs[[index_abs[i, 0]], :]\
                            for i in range(index_abs.shape[0])],
                            axis=0)
    index_inc = np.concatenate(
                    [[[index_abs[i, 0]-i, index_abs[i, 1]-i-1]]\
                    for i in range(index_abs.shape[0])],
                    axis=0)
    print 'data_inc.shape = {}'.format(data_inc.shape)
    print 'inits.shape = {}'.format(inits.shape)
    # print index_inc
    return inits, data_inc, index_inc

def inc2abs_forall(inits, data_inc, index_inc):
    data_abs = np.concatenate(
            [inc2abs(inits[i, :], data_inc[index_inc[i, 0]:index_inc[i, 1], :])\
            for i in range(index_inc.shape[0])],
            axis=0)
    index_abs = np.concatenate(
                                [[index_inc[i, 0]+i, index_inc[i, 1]+i+1]\
                                for i in range(index_inc.shape[0])],
                                axis=0)
    return data_abs, index_abs

