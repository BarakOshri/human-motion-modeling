def joint_pos(pos, idx):
    """
    Return the position of the idx'th joint. 
    """
    return [pos[idx*3+0], pos[idx*3+1], pos[idx*3+2]]

def joint_ori(ori, idx):
    """
    Return the orientation matrix of the idx'th joint. 
    """
    return [[ori[idx*3+0], ori[idx*3+1], ori[idx*3+2]], 
            [ori[idx*3+3], ori[idx*3+4], ori[idx*3+5]], 
            [ori[idx*3+6], ori[idx*3+7], ori[idx*3+8]]]

def get_limbs_length(connect