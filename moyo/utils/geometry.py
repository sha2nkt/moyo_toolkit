# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2021 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import torch
import cv2
import numpy as np
import math
import torch.nn.functional as F

def ea2rm(x, y, z):
    cos_x, sin_x = torch.cos(x), torch.sin(x)
    cos_y, sin_y = torch.cos(y), torch.sin(y)
    cos_z, sin_z = torch.cos(z), torch.sin(z)

    R = torch.stack(
            [torch.cat([cos_y*cos_z, sin_x*sin_y*cos_z - cos_x*sin_z, cos_x*sin_y*cos_z + sin_x*sin_z], dim=1),
            torch.cat([cos_y*sin_z, sin_x*sin_y*sin_z + cos_x*cos_z, cos_x*sin_y*sin_z - sin_x*cos_z], dim=1),
            torch.cat([-sin_y, sin_x*cos_y, cos_x*cos_y], dim=1)], dim=1)
    return R

def quaternion_to_axis_angle(quaternions):
    """
    Convert rotations given as quaternions to axis/angle.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: torch.Tensor):
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(*batch_dim, 9), dim=-1)
    q_abs = _sqrt_positive_part(torch.stack([ 1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22,  1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22,],dim=-1,))
    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack([torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1), torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1), torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1), torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),],dim=-2,)
    # clipping is not important here; if q_abs is small, the candidate won't be picked
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None])
    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    bla= quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(*batch_dim, 4)
    return bla

def matrix_to_axis_angle(matrix):
    """
    Convert rotations given as rotation matrices to axis/angle.
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))

def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
):
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter: str):
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2



def matrix_to_euler_angles(matrix, convention: str):
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)

def axis_angle_to_quaternion(axis_angle):
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def axis_angle_to_matrix(axis_angle):
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

def rotmat_from_euler_angles(pitch, roll, yaw):
    x = pitch
    y = yaw
    z = roll
    cos_x, sin_x = torch.cos(x), torch.sin(x)
    cos_y, sin_y = torch.cos(y), torch.sin(y)
    cos_z, sin_z = torch.cos(z), torch.sin(z)

    R = torch.stack(
            [torch.cat([cos_y*cos_z, sin_x*sin_y*cos_z - cos_x*sin_z, cos_x*sin_y*cos_z + sin_x*sin_z], dim=1),
            torch.cat([cos_y*sin_z, sin_x*sin_y*sin_z + cos_x*cos_z, cos_x*sin_y*sin_z - sin_x*cos_z], dim=1),
            torch.cat([-sin_y, sin_x*cos_y, cos_x*cos_y], dim=1)], dim=1)

    return R

def euler_angles_from_rotmat(R):
    """
    computer euler angles for rotation around x, y, z axis
    from rotation amtrix
    R: 4x4 rotation matrix
    https://www.gregslabaugh.net/publications/euler.pdf
    """
    r21 = np.round(R[:,2,0].item(), 4)
    if abs(r21) != 1:
        y_angle1 = -1 * torch.asin(R[:,2,0])
        y_angle2 = math.pi + torch.asin(R[:,2,0])
        cy1, cy2 = torch.cos(y_angle1), torch.cos(y_angle2)

        x_angle1 = torch.atan2(R[:,2,1] / cy1, R[:,2,2] / cy1)
        x_angle2 = torch.atan2(R[:,2,1] / cy2, R[:,2,2] / cy2)
        z_angle1 = torch.atan2(R[:,1,0] / cy1, R[:,0,0] / cy1)
        z_angle2 = torch.atan2(R[:,1,0] / cy2, R[:,0,0] / cy2)

        s1 = (x_angle1, y_angle1, z_angle1)
        s2 = (x_angle2, y_angle2, z_angle2)
        s = (s1, s2)

    else:
        z_angle = torch.tensor([0], device=R.device).float()
        if r21 == -1:
            y_angle = torch.tensor([math.pi / 2], device=R.device).float()
            x_angle = z_angle + torch.atan2(R[:,0,1], R[:,0,2])
        else:
            y_angle = -torch.tensor([math.pi / 2], device=R.device).float()
            x_angle = -z_angle + torch.atan2(-R[:,0,1], R[:,0,2])
        s = ((x_angle, y_angle, z_angle), )
    return s



def estimate_translation_np_old(S, joints_2d, joints_conf, focal_length_x=5000, focal_length_y=5000, W=224, H=224):
    """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Taken from: https://github.com/nkolot/SPIN/blob/master/utils/geometry.py
    Input:
        S: (25, 3) 3D joint locations
        joints: (25, 3) 2D joint locations and confidence
    Returns:
        (3,) camera translation vector
    """
    num_joints = S.shape[0]
    # focal length
    f = np.array([focal_length_x, focal_length_y])
    # optical center
    center = np.array([W/2., H/2.])
    # 2d joints mean
    p2d = joints_2d.mean(0)
    # 3d joints mean
    P3d = S.mean(0)[:2]

    trans_xy = (p2d-center) / f - P3d

    # transformations
    Z = np.reshape(np.tile(S[:,2],(2,1)).T,-1)
    XY = np.reshape(S[:,0:2],-1)
    O = np.tile(center,num_joints)
    F = np.tile(f,num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf),(2,1)).T,-1)

    # least squares
    Q = np.array([F*np.tile(np.array([1,0]),num_joints),
        F*np.tile(np.array([0,1]),num_joints), O-np.reshape(joints_2d,-1)]).T
    c = (np.reshape(joints_2d,-1)-O)*Z - F*XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W,Q)
    c = np.dot(W,c)

    # square matrix
    A = np.dot(Q.T,Q)
    b = np.dot(Q.T,c)

    # solution
    trans = np.linalg.solve(A, b)
    #trans[:2] = trans_xy

    return trans


def euler_angles_from_rotmat(R):
    """
    computer euler angles for rotation around x, y, z axis
    from rotation amtrix
    R: 4x4 rotation matrix
    https://www.gregslabaugh.net/publications/euler.pdf
    """
    r21 = np.round(R[:,2,0].item(), 4)
    if abs(r21) != 1:
        y_angle1 = -1 * torch.asin(R[:,2,0])
        y_angle2 = math.pi + torch.asin(R[:,2,0])
        cy1, cy2 = torch.cos(y_angle1), torch.cos(y_angle2)

        x_angle1 = torch.atan2(R[:,2,1] / cy1, R[:,2,2] / cy1)
        x_angle2 = torch.atan2(R[:,2,1] / cy2, R[:,2,2] / cy2)
        z_angle1 = torch.atan2(R[:,1,0] / cy1, R[:,0,0] / cy1)
        z_angle2 = torch.atan2(R[:,1,0] / cy2, R[:,0,0] / cy2)

        s1 = (x_angle1, y_angle1, z_angle1)
        s2 = (x_angle2, y_angle2, z_angle2)
        s = (s1, s2)

    else:
        z_angle = torch.tensor([0], device=R.device).float()
        if r21 == -1:
            y_angle = torch.tensor([math.pi / 2], device=R.device).float()
            x_angle = z_angle + torch.atan2(R[:,0,1], R[:,0,2])
        else:
            y_angle = -torch.tensor([math.pi / 2], device=R.device).float()
            x_angle = -z_angle + torch.atan2(-R[:,0,1], R[:,0,2])
        s = ((x_angle, y_angle, z_angle), )
    return s



def estimate_translation_np(S, joints_2d, joints_conf, 
    fx=5000, fy=5000, cx=112, cy=112, R=None, eqtype='optimize'):
    """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Taken from: https://github.com/nkolot/SPIN/blob/master/utils/geometry.py
    Input:
        S: (25, 3) 3D joint locations
        joints: (25, 3) 2D joint locations and confidence
    Returns:
        (3,) camera translation vector
    """


    num_joints = S.shape[0]
    # focal length
    f = np.array([fx, fy])
    # optical center
    center = np.array([cx, cy])
    # rotation
    if R is None:
        R = np.eye(3)

    if eqtype == 'iter1':
        # camera matrix 
        K = np.array(
            [[fx, 0, cx], 
            [0, fy, cy],
            [0, 0, 1]]
        )
        K_inv = np.linalg.inv(K)

        RS = np.einsum('ij,bi->bj', R, S)
        RS = RS[:,:2] / RS[:,2, np.newaxis]

        Kx = np.einsum('ij,bi->bj', K_inv[:2, :2], joints_2d) + K_inv[:2,2]

        Q = Kx - RS

        t = Q.sum(0) / num_joints

        # get scale factor 
        sfs = []
        for idx in range(joints_2d.shape[0]):
            sf = joints_2d[idx] / (K[:2, :2] @ ((R @ S[idx])[:2] + t) + K[:2, 2])
            sfs += [sf]
        z = np.array(sfs).mean()
        trans = np.array([t[0] * z, t[1] * z, 1 / z])

    elif eqtype == 'iter2':
        RS = np.einsum('ij,bi->bj', R, S)

        # transformations
        Z = np.reshape(np.tile(RS[:,2],(2,1)).T,-1)
        XY = np.reshape(RS[:,0:2],-1)
        O = np.tile(center,num_joints)
        F = np.tile(f,num_joints)
        weight2 = np.reshape(np.tile(np.sqrt(joints_conf),(2,1)).T,-1)

        # least squares
        Q = np.array([F*np.tile(np.array([1,0]),num_joints),
            F*np.tile(np.array([0,1]),num_joints), O-np.reshape(joints_2d,-1)]).T
        c = (np.reshape(joints_2d,-1)-O)*Z - F*XY

        # weighted least squares
        W = np.diagflat(weight2)
        Q = np.dot(W,Q)
        c = np.dot(W,c)

        # square matrix
        A = np.dot(Q.T,Q)
        b = np.dot(Q.T,c)

        # solution
        trans = np.linalg.solve(A, b)

    else:
        # transformations
        Z = np.reshape(np.tile(S[:,2],(2,1)).T,-1)
        XY = np.reshape(S[:,0:2],-1)
        O = np.tile(center,num_joints)
        F = np.tile(f,num_joints)
        weight2 = np.reshape(np.tile(np.sqrt(joints_conf),(2,1)).T,-1)

        # least squares
        Q = np.array([F*np.tile(np.array([1,0]),num_joints),
            F*np.tile(np.array([0,1]),num_joints), O-np.reshape(joints_2d,-1)]).T
        c = (np.reshape(joints_2d,-1)-O)*Z - F*XY

        # weighted least squares
        W = np.diagflat(weight2)
        Q = np.dot(W,Q)
        c = np.dot(W,c)

        # square matrix
        A = np.dot(Q.T,Q)
        b = np.dot(Q.T,c)

        # solution
        trans = np.linalg.solve(A, b)

    return trans

def optimize_translation(S, joints_2d, joints_conf, 
    fx=5000, fy=5000, cx=112, cy=112, R=None, eqtype='optimize'):
    """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Taken from: https://github.com/nkolot/SPIN/blob/master/utils/geometry.py
    Input:
        S: (25, 3) 3D joint locations
        joints: (25, 3) 2D joint locations and confidence
    Returns:
        (3,) camera translation vector
    """

    trans = None

    return trans

def estimate_translation_np_(S, joints_2d, joints_conf, focal_length_x=5000, focal_length_y=5000, W=224, H=224):
    """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Taken from: https://github.com/nkolot/SPIN/blob/master/utils/geometry.py
    Input:
        S: (25, 3) 3D joint locations
        joints: (25, 3) 2D joint locations and confidence
    Returns:
        (3,) camera translation vector
    """
    num_joints = S.shape[0]
    # focal length
    f = np.array([focal_length_x, focal_length_y])
    # optical center
    center = np.array([W/2., H/2.])
    # 2d joints mean
    p2d = joints_2d.mean(0)
    # 3d joints mean
    P3d = S.mean(0)[:2]

    trans_xy = (p2d-center) / f - P3d

    # transformations
    Z = np.reshape(np.tile(S[:,2],(2,1)).T,-1)
    XY = np.reshape(S[:,0:2],-1)
    O = np.tile(center,num_joints)
    F = np.tile(f,num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf),(2,1)).T,-1)

    # least squares
    Q = np.array([F*np.tile(np.array([1,0]),num_joints),
        F*np.tile(np.array([0,1]),num_joints), O-np.reshape(joints_2d,-1)]).T
    c = (np.reshape(joints_2d,-1)-O)*Z - F*XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W,Q)
    c = np.dot(W,c)

    # square matrix
    A = np.dot(Q.T,Q)
    b = np.dot(Q.T,c)

    # solution
    trans = np.linalg.solve(A, b)
    #trans[:2] = trans_xy

    return trans