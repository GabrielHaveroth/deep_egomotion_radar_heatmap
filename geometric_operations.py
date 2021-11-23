import numpy as np
import math
from transformations import *


def cartesian_to_spherical_coordinates(point_cartesian: np.array):
    delta_l = np.linalg.norm(point_cartesian)
    if np.absolute(delta_l) > 1e-05:
        theta = np.arccos(point_cartesian[2] / delta_l)
        psi = np.arctan2(point_cartesian[1], point_cartesian[0])
        return delta_l, theta, psi
    else:
        return 0, 0, 0


def is_rotation_matrix(R: np.array):
    """ 
        Checks if a matrix is a valid rotation matrix
        referred from https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotation_matrix_to_euler_angles(R: np.array):
    """ calculates rotation matrix to euler angles
        referred from https://www.learnopencv.com/rotation-matrix-to-euler-angles
    """
    assert(is_rotation_matrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])

# reorthogonalize the SO(3) part of SE(3) by normalizing a quaternion


def reorthogonalize_SE3(T: np.array):
    # ensure the rotational matrix is orthogonal
    q = transformations.quaternion_from_matrix(T)
    n = np.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
    q = q / n
    T_new = transformations.quaternion_matrix(q)
    T_new[0:3, 3] = T[0:3, 3]
    return T_new


def rotation_matrix_to_quaternion(R):
    assert (is_rotation_matrix(R))

    qw = np.sqrt(1 + np.sum(np.diag(R))) / 2.0
    qx = (R[2, 1] - R[1, 2]) / (4 * qw)
    qy = (R[0, 2] - R[2, 0]) / (4 * qw)
    qz = (R[1, 0] - R[0, 1]) / (4 * qw)

    return np.array([qw, qx, qy, qz])


def get_ground_6d_poses_euler(SE1, SE2):
    """ For 6dof pose representaion """

    SE12 = np.matmul(np.linalg.inv(SE1), SE2)
    SE12 = reorthogonalize_SE3(SE12)
    pos = np.array([SE12[0][3], SE12[1][3], SE12[2][3]])
    angles = rotation_matrix_to_euler_angles(SE12[:3, :3])
    return np.concatenate((angles, pos)), SE12   # rpyxyz


def get_ground_6d_poses_quat(SE1, SE2):
    """ For 6dof pose representaion """

    SE12 = np.matmul(np.linalg.inv(SE1), SE2)

    pos = np.array([SE12[0][3], SE12[1][3], SE12[2][3]])
    quat = rotation_matrix_to_quaternion(SE12[:3, :3])
    return np.concatenate((quat, pos)), SE12    # qxyz


def get_6D_poses_from_matrix(SE):
    pos = np.array([SE[0][3], SE[1][3], SE[2][3]])
    angles = rotation_matrix_to_euler_angles(SE[:3, :3])
    return np.concatenate((angles, pos))


def get_matrix_from_6D_relative_pose(angles, pos):
    T12 = euler_matrix(angles[0], angles[1], angles[2])
    T12[:3, 3] = pos
    return T12



"""
# Example of usage
>>> alpha1, beta1, gamma1 = 0.123, -1.234, 2.34
alpha2, beta2, gamma2 = 0.130, -1.134, 3.14
# Get rotation matrix
>>> T1 = euler_matrix(alpha1, beta1, gamma1)
>>> T2 = euler_matrix(alpha2, beta2, gamma2)
# Put on it a translation
>>> T1[:3, 3] = np.array([1.3, 2.8, 0.9])
>>> T2[:3, 3] = np.array([2, 3, 1])
# Calculating delta pose
>>> delta_pose, T12 = get_ground_6d_poses_euler(T1, T2)
>>> T12 = reorthogonalize_SE3(T12)
# Calculating the pose in relation the initial state
>>> abs_pose = reorthogonalize_SE3(.) 
# abs_pose[:3, 3] = np.array([0, 0, 0])
>>> print(euler_from_matrix(T1.dot(T12))) """
