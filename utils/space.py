# Space Transformation Utility
import numpy
from Quaternion import Quat

MACHINE_PRECISION = numpy.finfo(float).eps

def r3_to_quat(v):
    """
    Exponential map: Tranform vector in R^3 into a quatertion in S^3:

    exp([0, 0, 0]^T) = [0, 0, 0, 1]^T

    theta = ||v||
    v_hat = v / ||v||
    exp(v)  = [sin(1/2 * theta) / theta * v_hat, cos(1/2 * theta)] ^T
            = [1/2 + theta^2 /48 * v, cos(1/2 * theta)] ^T
    """
    if numpy.array_equal(v, [0, 0, 0]):
        quat = Quat((0, 0, 0, 1))
    else:
        theta = numpy.linalg.norm(v)
        # print 'r3_to_quat:theta: {}'.format(theta)  # debug

        if theta <= MACHINE_PRECISION:
            s = 1/2 + theta**2 / 48
        else:
            s = numpy.sin(0.5 * theta) / theta

        quat = Quat((s*v[0], s*v[1], s*v[2], numpy.cos(0.5 * theta)))
    return quat

def quat_to_r3(quat):
    """
    Inverse Exponential Map: Tranform quaternion in S^3 into vector in R^3:

    theta = 2 * arccos(quat.q[3])
    v = quat.q[0:3] / sin(1/2 * theta) * theta
    """
    theta = 2 * numpy.arccos(quat.q[3])
    # print 'quat_to_r3:theta: {}'.format(theta)  # debug
    v = quat.q[0:3] / numpy.sin(0.5 * theta) * theta
    return v

def r3_to_rmat(v):
    """
    Convert exponential map vector to rotation matrix (via quaternion). 
    """
    return r3_to_quat(v)._quat2transform()

def rmat_to_r3(rmat):
    """
    Convert rotation matrix to exponential map vector (via quaternion). 
    """
    return quat_to_r3(Quat(rmat))
    
def rotate(x, R):
    return R.dot(x)

def translate(x, y):
    return x + y

def pos_transform(x_w, x_space, R_space):
    """
    Transform a position in world(parent) space to object(child) space. 

    Parameters
    ----------
    x_w: \in R^3
        Position in world space. 
    x_space: \in R^3
        Origin position of the object space. 
    R_space: \in R^3 X R^3
        Orientation matrix of the object space. 

    Returns
    -------
    x_o: \in R^3 
        Position in object space
    """ 
    return numpy.dot(inv(R), (x_w - w_space))

def pos_inv_transform(x_o, x_space, R_space):
    """
    Inverse-transform a position in object(child) space to world(parent) space. 

    Parameters
    ----------
    x_o: \in R^3 
        Position in object space
    x_space: \in R^3
        Origin position of the object space. 
    R_space: \in R^3 X R^3
        Orientation matrix of the object space. 

    Returns
    -------
    x_w: \in R^3
        Position in world space. 
    """ 
    return x_space + numpy.dot(R, x_o)


def change_space(pos, ori, t = numpy.zeros(3), R = numpy.eye(3)):
    """
    Changing position and orientation arrays into a new coordinate system.

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
            pos_new[i, j*3:(j+1)*3] = numpy.dot(invR, (pos[i, j*3:(j+1)*3] - t))

    for j in range(ori.shape[1]/9):
        ori_new[i, j*9:(j+1)*9] = (invR * ori[i, j*9:(j+1)*9].reshape(3, 3)).reshape(1, 9)

    return pos_new, ori_new
