from typing import List

import numpy as np
import pytorch3d
import torch
import trimesh
from dl_ext.primitive import safe_zip
from pytorch3d.structures import Meshes

from easyhec.structures.nvdiffrast_renderer import NVDiffrastRenderer
from easyhec.structures.sapien_kin import SAPIENKinematicsModelStandalone
from easyhec.utils.utils_3d import matrix_3x4_to_4x4, create_center_radius, transform_points
from easyhec.structures.robot_mapping import robot_mapping

class NVdiffrastRenderMeshApiHelper:
    _renderer = None
    H, W = None, None

    @staticmethod
    def get_renderer(H, W):
        if NVdiffrastRenderMeshApiHelper._renderer is None or H != NVdiffrastRenderMeshApiHelper.H or W != NVdiffrastRenderMeshApiHelper.W:
            NVdiffrastRenderMeshApiHelper._renderer = NVDiffrastRenderer((H, W))
        return NVdiffrastRenderMeshApiHelper._renderer


def nvdiffrast_render_mesh_api(mesh: trimesh.Trimesh, object_pose, H, W, K, anti_aliasing=True):
    """
    :param mesh: trimesh mesh
    :param object_pose: object pose in camera coordinate
    :param H: image height
    :param W: image width
    :param K: camera intrinsics
    :return: mask
    """
    renderer = NVdiffrastRenderMeshApiHelper.get_renderer(H, W)
    verts = torch.from_numpy(mesh.vertices).float().cuda()
    faces = torch.from_numpy(mesh.faces).int().cuda()
    K = torch.from_numpy(K).float().cuda()
    object_pose = torch.from_numpy(object_pose).float().cuda()
    mask = renderer.render_mask(verts, faces, K, object_pose, anti_aliasing=anti_aliasing)
    mask = mask.cpu().numpy().astype(bool)
    return mask


def nvdiffrast_render_meshes_api(meshes: List[trimesh.Trimesh], object_poses, H, W, K, return_ndarray=True):
    """
    :param meshes: list of trimesh mesh
    :param object_poses: list of object poses in camera coordinate
    :param H: image height
    :param W: image width
    :param K: camera intrinsics
    :return: mask np.array of shape (H, W), bool, 0 or 1
    """
    renderer = NVdiffrastRenderMeshApiHelper.get_renderer(H, W)
    masks = []
    K = torch.from_numpy(K).float().cuda()
    for mesh, object_pose in safe_zip(meshes, object_poses):
        verts = torch.from_numpy(mesh.vertices).float().cuda()
        faces = torch.from_numpy(mesh.faces).int().cuda()
        object_pose = torch.from_numpy(object_pose).float().cuda()
        mask = renderer.render_mask(verts, faces, K, object_pose, anti_aliasing=False)
        masks.append(mask)
    mask = torch.stack(masks).float().sum(0).clamp(max=1)
    if return_ndarray:
        mask = mask.cpu().numpy().astype(bool)
    return mask


def nvdiffrast_parallel_render_meshes_api(meshes: List[trimesh.Trimesh], object_poses, H, W, K, return_ndarray=True):
    """
    :param meshes: list of trimesh mesh
    :param object_poses: list of object poses in camera coordinate
    :param H: image height
    :param W: image width
    :param K: camera intrinsics
    :return: mask np.array of shape (H, W), bool, 0 or 1
    """
    renderer = NVdiffrastRenderMeshApiHelper.get_renderer(H, W)
    K = torch.from_numpy(K).float().cuda()
    object_poses = torch.from_numpy(np.stack(object_poses)).float().cuda()
    verts_list = []
    faces_list = []
    for mesh, object_pose in safe_zip(meshes, object_poses):
        verts = torch.from_numpy(mesh.vertices).float().cuda()
        faces = torch.from_numpy(mesh.faces).int().cuda()
        v = transform_points(verts, object_pose)
        verts_list.append(v)
        faces_list.append(faces)
    mesh = pytorch3d.structures.Meshes(verts=verts_list, faces=faces_list)
    verts, faces = mesh.verts_packed(), mesh.faces_packed().int()
    masks = renderer.batch_render_mask(verts, faces, K, anti_aliasing=False)
    mask = masks.float()
    if return_ndarray:
        mask = mask.cpu().numpy().astype(bool)
    return mask


class RenderXarmApiHelper:
    meshes = None
    sk = None
    _urdf_path = None

    @staticmethod
    def get_meshes(robot_type = 0):
        if RenderXarmApiHelper.meshes is None:
            RenderXarmApiHelper.meshes = {}
            if robot_mapping[robot_type] == "Franka":
                from easyhec.structures.franka_mapping import link_name_mesh_path_mapping
            elif robot_mapping[robot_type] == "Xarm":
                from easyhec.structures.xarm_mapping import link_name_mesh_path_mapping
            for k, v in link_name_mesh_path_mapping.items():
                if v != "":
                    RenderXarmApiHelper.meshes[k] = trimesh.load(v, force="mesh")
        return RenderXarmApiHelper.meshes

    @staticmethod
    def get_sk(urdf_path):
        if RenderXarmApiHelper.sk is None or urdf_path != RenderXarmApiHelper._urdf_path:
            RenderXarmApiHelper.sk = SAPIENKinematicsModelStandalone(urdf_path)
            RenderXarmApiHelper._urdf_path = urdf_path
        return RenderXarmApiHelper.sk


def nvdiffrast_render_xarm_api(urdf_path, robot_pose, qpos, H, W, K, robot_type = 1, return_ndarray=True):
    xarm_meshes = RenderXarmApiHelper.get_meshes(robot_type=robot_type)
    sk = RenderXarmApiHelper.get_sk(urdf_path)
    names = [link.name for link in sk.robot.get_links()]
    poses = []
    meshes = []
    num = 8 if robot_mapping[robot_type] == "Xarm" else 7
    for i in range(num):
        pose = robot_pose @ sk.compute_forward_kinematics(qpos, i + 1).to_transformation_matrix()
        mesh = xarm_meshes[names[i + 1]]
        meshes.append(mesh)
        poses.append(pose)
    mask = nvdiffrast_render_meshes_api(meshes, poses, H, W, K, return_ndarray=return_ndarray)
    return mask


def nvdiffrast_parallel_render_xarm_api(urdf_path, robot_pose, qpos, H, W, K, robot_type = 1, return_ndarray=True):
    xarm_meshes = RenderXarmApiHelper.get_meshes(robot_type=robot_type)
    sk = RenderXarmApiHelper.get_sk(urdf_path)
    names = [link.name for link in sk.robot.get_links()]
    poses = []
    meshes = []
    num = 8 if robot_mapping[robot_type] == "Xarm" else 7
    for i in range(num):
        pose = robot_pose @ sk.compute_forward_kinematics(qpos, i + 1).to_transformation_matrix()
        mesh = xarm_meshes[names[i + 1]]
        meshes.append(mesh)
        poses.append(pose)
    mask = nvdiffrast_parallel_render_meshes_api(meshes, poses, H, W, K, return_ndarray=return_ndarray)
    return mask


def get_ring_object_poses(min_dist, max_dist, min_elev=-80, max_elev=80, ndist=5, nelev=18, nazim=12,
                          trans_noise=0.0, endpoint=True, start_azim=0, end_azim=2 * np.pi):
    """
    :param min_dist:
    :param max_dist:
    :param min_elev:
    :param max_elev:
    :param ndist:
    :param nelev:
    :param nazim:
    :return: object poses in camera coordinate
    """
    elevs = np.linspace(min_elev, max_elev, nelev)
    dists = np.linspace(min_dist, max_dist, ndist)
    all_RT = []
    for d in dists:
        for e in elevs:
            RT = torch.from_numpy(create_center_radius(dist=d, nrad=nazim, angle_z=e, endpoint=endpoint,
                                                       start=start_azim, end=end_azim))
            all_RT.append(RT)
    all_RT = torch.cat(all_RT, 0)
    poses = matrix_3x4_to_4x4(all_RT)
    noise = torch.randn([poses.shape[0], 3]) * torch.tensor(trans_noise)
    poses[:, :3, 3] = poses[:, :3, 3] + noise
    return poses
