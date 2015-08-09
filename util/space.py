# Space Transformation Utility
import numpy as np
from Quaternion import Quat

MACHINE_PRECISION = np.finfo(float).eps

def r3_to_quat(v):
    """
    Exponential map: Tranform vector in R^3 into a quatertion in S^3:

    exp([0, 0, 0]^T) = [0, 0, 0, 1]^T

    theta = ||v||
    v_hat = v / ||v||
    exp(v)  = [sin(1/2 * theta) / theta * v_hat, cos(1/2 * theta)] ^T
            = [1/2 + theta^2 /48 * v, cos(1/2 * theta)] ^T
    """
    if np.array_equal(v, [0, 0, 0]):
        quat = Quat((0, 0, 0, 1))
    else:
        theta = np.linalg.norm(v)
        # print 'r3_to_quat:theta: {}'.format(theta)  # debug

        if theta <= MACHINE_PRECISION:
            s = 1/2 + theta**2 / 48
        else:
            s = np.sin(0.5 * theta) / theta

        quat = Quat((s*v[0], s*v[1], s*v[2], np.cos(0.5 * theta)))
    return quat

def quat_to_r3(quat):
    """
    Inverse Exponential Map: Tranform quaternion in S^3 into vector in R^3:

    theta = 2 * arccos(quat.q[3])
    v = quat.q[0:3] / sin(1/2 * theta) * theta
    """
    theta = 2 * np.arccos(quat.q[3])
    # print 'quat_to_r3:theta: {}'.format(theta)  # debug
    v = quat.q[0:3] / np.sin(0.5 * theta) * theta
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
    return np.dot((x_w - x_space), R_space.T)

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
    return x_space + np.dot(x_o, R_space)

def xyz2sph(v):
    """
    Get the spherical coordinates from Cartisian coordinates. 
    """
    r = np.linalg.norm(v, ord=2)
    theta = np.arccos(v[2]/r)
    phi = np.arctan2(v[1], v[0])
    return [r, theta, phi]

def sph2xyz(r, theta, phi):
    """
    Get the Cartisian coordinates from spherical coordinates. 
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return [x, y, z]
    
def get_child_axises(vc_w, vp_w, ux_p, uy_p, uz_p):
    t = vc_w - vp_w
    uz_c = -t / np.linag.norm(t)
    ux_p_project = ux_p - np.dot(ux_p, uz_c)
    ux_c = ux_p_project / np.linag.norm(ux_p_project)
    uy_c = np.cross(uz_c, ux_c)
    return [ux_c, uy_c, uz_c]


