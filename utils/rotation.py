# Rotation Utility
import numpy
from Quaternion import Quat

MACHINE_PRECISION = numpy.finfo(float).eps

def r3_to_quat(v):
    """
    Exponential map from vector in R^3 to quaternion in S^3. 
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
        print 'r3_to_quat:theta: {}'.format(theta)  # debug

        if theta <= MACHINE_PRECISION:
            s = 1/2 + theta**2 / 48
        else:
            s = numpy.sin(0.5 * theta) / theta

        quat = Quat((s*v[0], s*v[1], s*v[2], numpy.cos(0.5 * theta)))
    return quat


def quat_to_r3(quat):
    """
    Reverse xponential map from vector in R^3 to quaternion in S^3. 
    theta = 2 * arccos(quat.q[3])
    v = quat.q[0:3] / sin(1/2 * theta) * theta
    """
    theta = 2 * numpy.arccos(quat.q[3])
    print 'quat_to_r3:theta: {}'.format(theta)  # debug
    v = quat.q[0:3] / numpy.sin(0.5 * theta) * theta
    return v

