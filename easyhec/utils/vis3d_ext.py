import loguru
import matplotlib.pyplot as plt
import json
from typing import Union

import os
import shutil
import warnings

import PIL.Image
import numpy as np
import torch
import transforms3d
import trimesh.primitives
import wis3d
from matplotlib.cm import get_cmap
from scipy.spatial.transform import Rotation
from transforms3d import euler, affines
from wis3d.wis3d import tensor2ndarray, folder_names, file_exts

from easyhec.utils.comm import get_rank
from easyhec.utils.os_utils import magenta
from easyhec.utils.pn_utils import random_choice, to_array, clone_if_present, numel
from wis3d import Wis3D


class Vis3D(Wis3D):
    # ensure out_folder will be deleted once only when program starts.
    has_removed = []
    default_xyz_pattern = ('x', 'y', 'z')
    sequence_ids = {}
    default_out_folder = 'dbg'
    default_colors = {
        'orange': [249, 209, 22],
        'pink': [255, 63, 243],
        # 'green': [0, 255, 72],
        'green': [0, 255, 0],
        'blue': [2, 83, 255]
    }
    _xarm_sk = None

    def __init__(self, xyz_pattern=None, out_folder='dbg',
                 sequence='sequence',
                 auto_increase=True,
                 enable: bool = True):
        assert enable in [True, False]
        self.enable = enable and get_rank() == 0
        # loguru.logger.info(f'{get_rank()},enable,{self.enable}')
        if enable is True:
            if xyz_pattern is None:
                xyz_pattern = Vis3D.default_xyz_pattern
            # if out_folder is None:
            #     out_folder = Vis3D.default_out_folder
            if not os.path.isabs(out_folder):
                seq_out_folder = os.path.join(
                    os.getcwd(), out_folder, sequence)
            else:
                seq_out_folder = out_folder
            if os.path.exists(seq_out_folder) and seq_out_folder not in Vis3D.has_removed:
                shutil.rmtree(seq_out_folder)
                Vis3D.has_removed.append(seq_out_folder)
            super().__init__(out_folder, sequence, xyz_pattern)

            if seq_out_folder not in Vis3D.sequence_ids:
                Vis3D.sequence_ids[seq_out_folder] = 0
            else:
                Vis3D.sequence_ids[seq_out_folder] += 1
            self.auto_increase = auto_increase
            if auto_increase:
                scene_id = Vis3D.sequence_ids[seq_out_folder]
            else:
                scene_id = 0
            print(magenta(f'Set up Vis3D for {sequence}: {scene_id}'))
            # self.set_scene_id(scene_id)
            super().set_scene_id(scene_id)
            self.plane_model = None

    def add_point_cloud_sdf(self, points, sdf, truncation=1.0, sample=1.0, name=None):
        if not self.enable:
            return
        if sample < 1.0:
            sample_size = int(points.shape[0] * sample)
            _, idxs = random_choice(points, sample_size, dim=0)
            points = points[idxs]
            sdf = sdf[idxs]
        cmap = get_cmap('jet')
        assert truncation >= 0
        sdf = np.clip(to_array(sdf), a_min=-truncation, a_max=truncation)
        # colors = cmap(sdf)[:, :3]
        b = (sdf - np.min(sdf)) / np.ptp(sdf)
        colors = cmap(b)[:, :3]
        colors = (colors * 255).astype(np.uint8)
        self.add_point_cloud(points, colors=colors, name=name)

    def add_lines(self, start_points, end_points, colors=None, name=None):
        if not self.enable:
            return
        super().add_lines(start_points, end_points, colors=colors, name=name)

    def add_rays(self, rays_o, rays_d, max=10, min=0, sample=1.0, name=None):
        """[summary]

        Args:
            rays_o ([type]): [description]
            rays_d ([type]): [description]
            length (int, optional): [description]. Defaults to 10.
            sample (float, optional): [description]. Defaults to 1.0.
            name ([type], optional): [description]. Defaults to None.
        """
        if not self.enable:
            return
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        if sample < 1.0:
            size = int(sample * rays_d.shape[0])
            _, idx = random_choice(rays_o, size, dim=0)
            rays_o = rays_o[idx]
            rays_d = rays_d[idx]
        elif sample > 1.0:
            size = int(sample)
            _, idx = random_choice(rays_o, size, dim=0)
            rays_o = rays_o[idx]
            rays_d = rays_d[idx]

        self.add_lines(rays_o + rays_d * min, rays_o + rays_d * max, name=name)

    def add_3d_matching(self, points1, points2, matching,
                        sample=1.0, colors=None, name=None, cover_start_end=True):
        if not self.enable:
            return
        if np.size(points1) == 0:
            return
        perm = np.random.choice(np.arange(matching.shape[0]),
                                int(matching.shape[0] * sample),
                                replace=False)
        matching = matching[perm]
        pt1s = points1[matching[:, 0]]
        pt2s = points2[matching[:, 1]]
        if isinstance(colors, str):
            colors = np.array(self.default_colors[colors])[None].repeat(
                matching.shape[0], axis=0).astype(np.uint8)
        self.add_lines(pt1s, pt2s, colors=colors, name=name)

    def add_3d_matching_with_scores(self, points1, points2, matching, scores, sample=1.0, name=None):
        if not self.enable:
            return
        scores = to_array(scores)
        cmap = get_cmap('jet')
        scores = -np.log(1 - scores + 1e-6)
        colors = cmap(scores)[:, :3]
        colors = (colors * 255).astype(np.uint8)
        self.add_3d_matching(points1, points2, matching, sample, colors, name)

    def add_flow_3d(self, pts0, flow_3d, valid=None, min_percentile=0, max_percentile=99, sample=1.0, name=None):
        if not self.enable:
            return
        flow_3d = to_array(flow_3d)
        pts0 = to_array(pts0)
        if valid is None:
            valid = np.ones_like(flow_3d[..., 0]).astype(np.bool)
        valid = valid.reshape(-1).astype(np.bool)
        pts0 = pts0[valid]
        offsets = flow_3d.reshape(-1, 3)[valid]
        norms = np.linalg.norm(offsets, axis=1)
        assert min_percentile <= max_percentile
        high_thresh = np.percentile(norms, max_percentile)
        low_thresh = np.percentile(norms, min_percentile)
        keep = np.logical_and(norms > low_thresh, norms < high_thresh)
        pts0 = pts0[keep]
        offsets = offsets[keep]
        assert offsets.shape[0] == pts0.shape[0]
        pts1 = pts0 + offsets
        matching = np.arange(pts0.shape[0]).reshape(-1, 1)
        matching = np.hstack([matching, matching])
        # self.add_point_cloud(pts0, )
        # self.add_point_cloud(pts1, )
        self.add_3d_matching(pts0, pts1, matching, sample=sample, name=name)

    def add_box_by_6border(self, xmin, ymin, zmin, xmax, ymax, zmax, name=None):
        if not self.enable:
            return
        x = (xmin + xmax) / 2
        y = (ymin + ymax) / 2
        z = (zmin + zmax) / 2
        sx = xmax - xmin
        sy = ymax - ymin
        sz = zmax - zmin
        self.add_boxes(np.array([x, y, z]), np.array(
            [0, 0, 0]), np.array([sx, sy, sz]), name=name)

    def add_box_by_bounds(self, bounds):
        """

        :param bounds: 3x2 x,y,z
        :return:
        """
        if not self.enable:
            return
        bounds = to_array(bounds)
        xmin, ymin, zmin = bounds[:, 0]
        xmax, ymax, zmax = bounds[:, 1]
        self.add_box_by_6border(xmin, ymin, zmin, xmax, ymax, zmax)

    def add_boxes_by_dof(self, positions, rotations, scales, name=None, label=None):
        if not self.enable:
            return
        super().add_boxes(positions, rotations, scales, name, label)

    def add_point_cloud(self, points, colors=None, name=None, sample=1.0,
                        remove_plane=False, remove_plane_distance_thresh=0.005, remove_plane_cache_model=True,
                        max_z=100000.0, min_norm=0.0):
        if not self.enable:
            return
        points = to_array(points)
        if len(points.shape) != 2:
            points = points.reshape(-1, 3)
        if numel(points) == 0:
            return
        if sample < 1.0:
            points = torch.tensor(points)
            if points.ndim == 1:
                points = points.unsqueeze(0)
            sample_size = int(points.shape[0] * sample)
            if sample_size > 100:
                points, idxs = random_choice(points, sample_size, dim=0)
                if colors is not None:
                    colors = colors[idxs]
        elif sample > 1.0:
            points = torch.tensor(points)
            if points.ndim == 1:
                points = points.unsqueeze(0)
            sample_size = sample
            points, idxs = random_choice(points, sample_size, dim=0)
            if colors is not None:
                colors = colors[idxs]
        if max_z < 10000:
            keep = points[:, 2] < max_z
            points = points[keep]
            if colors is not None:
                colors = colors[keep]
        if min_norm > 0:
            norm = np.linalg.norm(points, axis=-1)
            keep = norm > min_norm
            points = points[keep]
            if colors is not None:
                colors = colors[keep]
        if remove_plane:
            from .utils_3d import open3d_plane_segment_api, point_plane_distance_api
            if not remove_plane_cache_model or self.plane_model is None:
                plane_model, inliers = open3d_plane_segment_api(points, remove_plane_distance_thresh)
                keep = np.ones([points.shape[0]], dtype=bool)
                keep[inliers] = 0
            else:
                dists = point_plane_distance_api(points, self.plane_model)
                keep = dists > remove_plane_distance_thresh
            points = points[keep]
            if colors is not None:
                colors = colors[keep]

        super().add_point_cloud(points, colors, name=name)

    def add_point_cloud_with_error(self, points, error, name=None):
        if not self.enable:
            return
        if points.numel() == 0:
            return
        self.add_point_cloud_sdf(points, error, name=name)

    @staticmethod
    def set_default_xyz_pattern(xyz_pattern):
        Vis3D.default_xyz_pattern = xyz_pattern

    def set_scene_id(self, id):
        if not self.enable:
            return
        if self.auto_increase:
            warnings.warn(
                "Auto-increase in ON. You should not set_scene_id manually.")
        super().set_scene_id(id)
        self.add_point_cloud(1000 * torch.ones([1, 3]), name='dummy')

    def add_camera_trajectory(self, poses: Union[np.ndarray, torch.Tensor], *, name: str = None) -> None:
        """
        Add a camera trajectory

        :param poses: transformation matrices of shape `(n, 4, 4)`

        :param name: output name of the camera trajectory
        """
        if not self.enable:
            return
        # super(Vis3D, self).add_camera_trajectory(poses, name=name)
        poses = tensor2ndarray(poses)

        poses = (self.three_to_world @ poses.T).T
        poses[:, :, [1, 2]] *= -1
        # r = Rotation.from_matrix(poses[:, :3, : 3])
        # eulers = r.as_euler('xyz')
        eulers = []
        positions = poses[:, :3, 3].reshape((-1, 3))
        for pose in poses:
            trans_euler = euler.mat2euler(pose[:3, :3], 'rxyz')
            # trans_euler = euler.mat2euler(pose[:3, :3])
            eulers.append(trans_euler)

        # print("eulers: ", eulers)
        # print("euler: ", eulers)

        filename = self.__get_export_file_name('camera_trajectory', name)
        with open(filename, 'w') as f:
            f.write(json.dumps(dict(eulers=eulers, positions=positions.tolist())))

    def add_image(self, image, name=None):
        if not self.enable:
            return
        if isinstance(image, (str, PIL.Image.Image)):
            super().add_image(image, name=name)
        elif isinstance(image, np.ndarray):
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = (image * 255).astype(np.uint8)
            image = PIL.Image.fromarray(image)
            self.add_image(image, name)
        elif isinstance(image, torch.Tensor):
            image = to_array(image)
            if image.shape[0] == 3:
                image = image.permute(1, 2, 0)
            self.add_image(image, name)
        else:
            raise TypeError()

    def add_spheres(self, centers, radius, colors=None, name=None):
        if not self.enable:
            return
        # centers = to_array(centers)
        # radius = float(radius)
        # vec = np.random.randn(3, n_points)
        # vec /= np.linalg.norm(vec, axis=0)
        # points = centers[..., None] + (vec * radius)[None, ...]
        # points = points.transpose(0, 2, 1).reshape(-1, 3)
        # self.add_point_cloud(points, colors=colors, name=name)
        super().add_spheres(centers, radius, colors=colors, name=name)

    def add_deformation_graph(self, graph, colors=None, name=None):
        import torchsparse as ts

        if not self.enable:
            return
        from easyhec.structures.deformation_graph import DeformationGraph
        assert isinstance(graph, DeformationGraph)
        graph = graph.numpy()
        edges = np.stack([np.repeat(np.arange(graph.graph_edges.shape[0])[:, None], 8, 1),
                          graph.graph_edges],
                         -1).reshape(-1, 2)
        edges = edges[edges[:, 1] != -1]
        self.add_graph(graph.node_positions, edges, colors=colors, name=name)
        # add warped version
        # warped_graph = graph.warp_itself()
        # edges = np.stack(
        #     [np.repeat(np.arange(warped_graph.graph_edges.shape[0])[:, None], 8, 1), warped_graph.graph_edges],
        #     -1).reshape(-1, 2)
        # warped_graph_name = None if name is None else name + "_warped"
        # self.add_graph(warped_graph.node_positions, edges, colors=colors, name=warped_graph_name)

    def add_mesh(self, vertices, faces=None, vertex_colors=None, *, name=None):
        if not self.enable:
            return
        if vertices is None:
            return
        vertices = clone_if_present(vertices)
        faces = clone_if_present(faces)
        vertex_colors = clone_if_present(vertex_colors)
        if isinstance(vertices, trimesh.Trimesh):
            vertex_colors = vertices.visual.vertex_colors
        super().add_mesh(vertices, faces, vertex_colors, name=name)

    def add_voxel(self, positions, voxel_size, colors=None, *, name=None):
        if not self.enable:
            return
        voxel_size = float(voxel_size)
        super(Vis3D, self).add_voxel(positions, voxel_size, colors, name=name)

    def add_depth_map(self, depth, name=None):
        cmap = get_cmap('jet')
        depth = cmap(to_array(depth))
        depth = ((depth[:, :, :3]) * 255).astype(np.uint8)
        self.add_image(depth, name=name)

    def add_plt(self, x, name=None, **kwargs):
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.gca()

        # ax.text(0.0, 0.0, "Test", fontsize=45)
        ax.axis('off')
        fig.tight_layout(pad=0)

        # To remove the huge white borders
        ax.margins(0)
        x = to_array(x)
        ax.imshow(x, **kwargs)
        # plt.axis('off')
        canvas.draw()  # draw the canvas, cache the renderer

        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        self.add_image(image, name=name)

    def increase_scene_id(self):
        if not self.enable:
            return
        # self.set_scene_id(self.scene_id + 1)
        super().set_scene_id(self.scene_id + 1)

    def add_flow2d(self, flow, name=None):
        """
        :param flow: h,w,2
        :return:
        """
        from easyhec.utils.flow_viz import flow_to_image
        flow_img = flow_to_image(flow)
        self.add_image(flow_img, name=name)

    def add_boxes(self, positions, eulers=None, extents=None, *, order=(0, 1, 2, 3, 4, 5, 6, 7), labels=None,
                  name=None):
        if not self.enable:
            return
        positions = tensor2ndarray(positions).copy()

        if eulers is None or extents is None:
            positions = np.asarray(positions).reshape(-1, 8, 3)
            corners = positions
            if order != (0, 1, 2, 3, 4, 5, 6, 7):
                for i, o in enumerate(order):
                    corners[:, o, :] = positions[:, i, :]

            positions = (corners[:, 0, :] + corners[:, 6, :]) / 2
            vector_xs = corners[:, 1, :] - corners[:, 0, :]
            vector_ys = corners[:, 4, :] - corners[:, 0, :]
            vector_zs = corners[:, 3, :] - corners[:, 0, :]

            extent_xs = np.linalg.norm(vector_xs, axis=1).reshape(-1, 1)
            extent_ys = np.linalg.norm(vector_ys, axis=1).reshape(-1, 1)
            extent_zs = np.linalg.norm(vector_zs, axis=1).reshape(-1, 1)
            extents = np.hstack((extent_xs, extent_ys, extent_zs))

            rot_mats = np.stack(
                (vector_xs / extent_xs, vector_ys / extent_ys, vector_zs / extent_zs), axis=2)
            Rs = Rotation.from_matrix(rot_mats)
            eulers = Rs.as_euler('XYZ')
        else:
            positions = tensor2ndarray(positions)
            eulers = tensor2ndarray(eulers)
            extents = tensor2ndarray(extents)
            positions = np.asarray(positions).reshape(-1, 3)
            extents = np.asarray(extents).reshape(-1, 3)
            eulers = np.asarray(eulers).reshape(-1, 3)

        boxes = []
        for i in range(len(positions)):
            box_def = self.three_to_world @ affines.compose(
                positions[i], euler.euler2mat(*eulers[i], 'rxyz'), extents[i])
            T, R, Z, _ = affines.decompose(box_def)
            box = dict(
                position=T.tolist(),
                euler=euler.mat2euler(R, 'rxyz'),
                extent=Z.tolist()
            )
            if labels is not None:
                if isinstance(labels, str):
                    labels = [labels]
                box.update({'label': labels[i]})

            boxes.append(box)

        filename = self.__get_export_file_name('boxes', name)
        with open(filename, 'w') as f:
            f.write(json.dumps(boxes))

    # def add_plt(self, x,name=None, **kwargs):
    #     plt.imshow(x, **kwargs)
    #     plt.gca()

    def __repr__(self):
        if not self.enable:
            return f'Vis3D:NA'
        else:
            return f'Vis3D:{self.sequence_name}:{self.scene_id}'

    def add_unit_cube(self):
        mesh = trimesh.primitives.Box()
        self.add_mesh(mesh)

    def add_plane(self, x=None, y=None, z=None, scale=10, name=None):
        assert (x is not None) + (y is not None) + (z is not None) == 1
        if x is not None:
            self.add_box_by_6border(x, -scale, -scale, x + 0.01, scale, scale, name=name)
        if y is not None:
            self.add_box_by_6border(-scale, y, -scale, scale, y + 0.01, scale, name=name)
        if z is not None:
            self.add_box_by_6border(-scale, -scale, z, scale, scale, z + 0.01, name=name)

    def __get_export_file_name(self, file_type: str, name: str = None) -> str:
        export_dir = os.path.join(
            self.out_folder,
            self.sequence_name,
            "%05d" % self.scene_id,
            folder_names[file_type],
        )
        os.makedirs(export_dir, exist_ok=True)
        if name is None:
            name = "%05d" % self.counters[file_type]

        filename = os.path.join(export_dir, name + "." + file_exts[file_type])
        self.counters[file_type] += 1

        return filename

    def add_volume(self, bounds, dimension):
        xmin, ymin, zmin = bounds[0]
        xmax, ymax, zmax = bounds[1]
        positions = torch.stack(torch.meshgrid(torch.linspace(xmin, xmax, dimension[0]),
                                               torch.linspace(ymin, ymax, dimension[1]),
                                               torch.linspace(zmin, zmax, dimension[2]), ), -1)
        positions = positions.reshape(-1, 3)
        positions, _ = random_choice(positions, 10000, dim=0, replace=False)
        self.add_voxel(positions, (xmax - xmin) / dimension[0])
        print()

    def add_xarm(self, qpos, Tw_w2B=None, add_local_coord=False, name=""):
        """

        Parameters
        ----------
        qpos: 7-dim ndarray, radian
        Tw_w2B: base in world
        add_local_coord
        name

        Returns
        -------

        """
        from easyhec.utils import utils_3d
        from easyhec.utils.utils_3d import Rt_to_pose
        if not self.enable: return
        if Tw_w2B is None:
            Tw_w2B = np.eye(4)
        qpos = to_array(qpos)
        # assert qpos.shape[0] == 7
        # qpos = qpos.tolist() + [0] * 6
        sk = self.xarm_sk
        links = sk.robot.get_links()
        num_links = len(links)
        # for link in range(num_links):
        #     pq = sk.compute_forward_kinematics(qpos, link)
        # print("link", sk.robot.get_links()[link], pq)
        # self.add_spheres(np.array(pq.p), 0.01, name=f'link{link:02d}_{sk.robot.get_links()[link].name}')
        local_pts = np.array([[0, 0, 0],
                              [1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]]) * 0.05
        for link in range(num_links):
            link_name = links[link].name

            pq = sk.compute_forward_kinematics(qpos, link)

            pose = Rt_to_pose(transforms3d.quaternions.quat2mat(pq.q), pq.p)
            from easyhec.structures.xarm_mapping import link_name_mesh_path_mapping
            mesh_path = link_name_mesh_path_mapping[link_name]
            if mesh_path == "": continue
            mesh = trimesh.load_mesh(mesh_path)
            add_name = link_name if name == "" else name + "_" + link_name
            self.add_mesh(utils_3d.transform_points(mesh.vertices, Tw_w2B @ pose), mesh.faces, name=add_name)
            axis_in_base = utils_3d.transform_points(local_pts, Tw_w2B @ pose)
            if add_local_coord:
                self.add_lines(axis_in_base[0], axis_in_base[1], name=f'{link_name}_x')
                self.add_lines(axis_in_base[0], axis_in_base[2], name=f'{link_name}_y')
                self.add_lines(axis_in_base[0], axis_in_base[3], name=f'{link_name}_z')
        # self.add_spheres(np.array([0.20600027, 0.05546901, 0.00436316]), 0.01, name='p_w')
        # self.add_spheres(np.array([0.20600033, -0.055461, 0.00436242]), 0.01, name='p_w2')

    @property
    def xarm_sk(self):
        if Vis3D._xarm_sk is None:
            from easyhec.structures.sapien_kin import SAPIENKinematicsModelStandalone
            urdf_path = os.path.abspath("assets/xarm7_with_gripper_reduced_dof.urdf")
            sk = SAPIENKinematicsModelStandalone(urdf_path)
            Vis3D._xarm_sk = sk
        return Vis3D._xarm_sk

    def add_text(self, text):
        if not self.enable: return
        self.add_image(np.zeros((1, 100, 3)), name=text)
