import numpy as np
import sapien.core as sapien
import transforms3d.quaternions


def pose4x4_to_sapien_pose(pose4x4):
    q = transforms3d.quaternions.mat2quat(pose4x4[:3, :3])
    p = pose4x4[:3, 3]
    return sapien.Pose(p, q)


def sapien_pose_to_pose4x4(sapien_pose):
    R = transforms3d.quaternions.quat2mat(sapien_pose.q)
    t = sapien_pose.p
    return Rt_to_pose(R, t)


def sapien_pose_to_7darray(sapien_pose):
    return np.concatenate([sapien_pose.p, sapien_pose.q])


def pose4x4_to_7darray(pose4x4):
    return sapien_pose_to_7darray(pose4x4_to_sapien_pose(pose4x4))


def Rt_to_pose(R, t=np.zeros(3)):
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t
    return pose
