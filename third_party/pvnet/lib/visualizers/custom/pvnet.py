import trimesh
import io
import os
import os.path as osp

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.coco as coco
import zarr
from PIL import Image
from easyhec.utils import utils_3d
from easyhec.utils.vis3d_ext import Vis3D
from easyhec.utils.utils_3d import depth_to_rect, transform_points, matrix_3x4_to_4x4
from termcolor import colored

from lib.config import cfg
from lib.datasets.dataset_catalog import DatasetCatalog
from lib.utils import img_utils
from lib.utils.pvnet import pvnet_config
from lib.utils.pvnet import pvnet_pose_utils


mean = pvnet_config.mean
std = pvnet_config.std


class Visualizer:

    def __init__(self):
        self._coco = None
        if cfg.test.icp or cfg.dbg:
            mesh: trimesh.Trimesh = trimesh.load_mesh(cfg.custom.model_path)
            mesh = mesh.apply_scale(cfg.custom.model_scale)
            # keep = mesh.vertices[:, 2] - mesh.vertices[:, 2].min() >= cfg.test.icp_trim
            # mesh.update_vertices(keep)
            # model = inout.load_ply(cfg.custom.model_path)
            model = {'pts': mesh.vertices,
                     'faces': mesh.faces}
            self.model = model
            model['pts'] = model['pts'] * 1000.0
            width = cfg.custom.width
            height = cfg.custom.height
            if cfg.test.icp:
                self.icp_refiner = icp_utils.ICPRefiner(model, (width, height))
        if cfg.smooth.enable:
            self.hist_pts = []
            self.smooth_num = cfg.smooth.num
            self.smooth_std = cfg.smooth.std
        self.vis3d = Vis3D(
            xyz_pattern=('x', '-y', '-z'),
            out_folder="dbg",
            sequence="custom_pvnet",
            # auto_increase=,
            # enable=,
        )

    def visualize(self, output, batch, show=True):
        inp = img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0)
        kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()

        img_id = int(batch['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        kpt_3d = np.concatenate([anno['fps_3d'], [anno['center_3d']]], axis=0)
        K = np.array(anno['K'])

        pose_gt = np.array(anno['pose'])
        pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)

        corner_3d = np.array(anno['corner_3d'])
        corner_2d_gt = pvnet_pose_utils.project(corner_3d, K, pose_gt)
        corner_2d_pred = pvnet_pose_utils.project(corner_3d, K, pose_pred)

        # _, ax = plt.subplots(1)
        plt.figure(dpi=200)
        plt.imshow(inp)
        plt.gca().add_patch(
            patches.Polygon(xy=corner_2d_gt[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='red'))
        plt.gca().add_patch(
            patches.Polygon(xy=corner_2d_gt[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='red'))
        plt.gca().add_patch(
            patches.Polygon(xy=corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='green'))
        plt.gca().add_patch(
            patches.Polygon(xy=corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='green'))
        plt.scatter(output['kpt_2d'][0].cpu()[:, 0], output['kpt_2d'][0].cpu()[:, 1], s=4, marker='*')
        for i in range(9):
            plt.text(output['kpt_2d'][0].cpu()[i, 0], output['kpt_2d'][0].cpu()[i, 1], str(i), fontsize=4, )
            # plt.scatter(output['kpt_2d'][0].cpu()[:, 0], output['kpt_2d'][0].cpu()[:, 1], s=4, marker='*')
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png")
        self.vis3d.add_image(np.array(Image.open(img_buf))[:, :, :3])
        self.vis3d.add_image(output['mask'][0].cpu(), name='mask')
        self.vis3d.increase_scene_id()
        plt.close("all")
        if show:
            plt.show()

    def visualize_train(self, output, batch):
        inp = img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0)
        mask = batch['mask'][0].detach().cpu().numpy()
        vertex = batch['vertex'][0][0].detach().cpu().numpy()
        img_id = int(batch['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        fps_2d = np.array(anno['fps_2d'])
        plt.figure(0)
        plt.subplot(221)
        plt.imshow(inp)
        plt.subplot(222)
        plt.imshow(mask)
        plt.plot(fps_2d[:, 0], fps_2d[:, 1])
        plt.subplot(224)
        plt.imshow(vertex)
        plt.show()
        print()
        # plt.savefig('test.jpg')
        # plt.close(0)

    def icp_refine(self, pose_pred, depth, output, K):
        depth = depth * 1000.0
        # mask = torch.argmax(output['seg'], dim=1)[0].detach().cpu().numpy()
        mask = output['mask'][0].detach().cpu().numpy()
        if pose_pred[2, 3] <= 0:
            print(colored(f"pose_pred[2,3]={pose_pred[2, 3]} < 0, skip!", "red"))
            return pose_pred
        depth[mask != 1] = 0
        pose_pred_tmp = pose_pred.copy()
        pose_pred_tmp[:3, 3] = pose_pred_tmp[:3, 3] * 1000.0
        R_refined, t_refined = self.icp_refiner.refine(depth, pose_pred_tmp[:3, :3], pose_pred_tmp[:3, 3], K.copy(),
                                                       depth_only=True, max_mean_dist_factor=5.0)
        R_refined, _ = self.icp_refiner.refine(depth, R_refined, t_refined, K.copy(), no_depth=True)
        if cfg.dbg:
            from easyhec.utils.vis3d_ext import Vis3D
            from easyhec.utils.utils_3d import depth_to_rect, transform_points, matrix_3x4_to_4x4
            vis3d = Vis3D(
                xyz_pattern=('x', '-y', '-z'),
                out_folder="dbg",
                sequence="icp_refine",
                # auto_increase=,
                # enable=,
            )
            fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            pc = depth_to_rect(fu, fv, cu, cv, depth / 1000.0)
            pc = pc[pc[:, 2] > 0]
            vis3d.add_point_cloud(pc)
            vis3d.add_mesh(transform_points(self.model['pts'] / 1000.0, matrix_3x4_to_4x4(pose_pred)),
                           self.model['faces'], name='pose_pred')
            tmp = np.hstack((R_refined, t_refined.reshape((3, 1)) / 1000))
            vis3d.add_mesh(transform_points(self.model['pts'] / 1000.0, matrix_3x4_to_4x4(tmp)),
                           self.model['faces'], name='pose_icp')
        pose_pred = np.hstack((R_refined, t_refined.reshape((3, 1)) / 1000))

        return pose_pred

    def visualize_demo(self, output, inp, meta, depth=None):
        if cfg.demo_vis_name == "":
            vis_dir = osp.join(cfg.model_dir, cfg.demo_dir.split("/")[-1], str(cfg.test.epoch))
        else:
            vis_dir = osp.join(cfg.model_dir, cfg.demo_vis_name, str(cfg.test.epoch))
        if cfg.test.icp:
            vis_dir = vis_dir + "_icp"
        if cfg.smooth.enable:
            vis_dir = vis_dir + "_smooth"
        os.makedirs(vis_dir, exist_ok=True)

        inp = img_utils.unnormalize_img(inp[0], mean, std).permute(1, 2, 0)
        if 'kpts_2d_anno' not in output or output['kpts_2d_anno'] is None:
            kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()
            kpt_3d = np.array(meta['kpt_3d'])
        else:
            kpt_2d = output['kpts_2d_anno']
            kpt_3d = np.array(meta['kpt_3d'])[cfg.demo_use_anno_idx]
            print("!!!!!")
            print("!using anno 2d!!")
            print("!!!!!")

        K = np.array(meta['K'])

        pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)
        if cfg.test.icp:
            pose_pred = self.icp_refine(pose_pred.copy(), depth, output, K)
            if cfg.test.icp_use_plane:
                plane_model = cfg.test.icp_plane_model
                # fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                # pts_rect = utils_3d.depth_to_rect(fu, fv, cu, cv, depth)
                # plane_model, inliers = utils_3d.open3d_plane_segment_api(pts_rect[pts_rect[:, 2] < 1.5], 0.01)
                # self.vis3d.add_point_cloud(pts_rect[pts_rect[:, 2] < 1.5][inliers], name='plane')
                # self.vis3d.add_point_cloud(pts_rect[pts_rect[:, 2] < 1.5], name='all_points')
                dist = utils_3d.point_plane_distance_api(pose_pred[:3, 3][None], np.array(plane_model))[0]
                print()
            kpt_2d = pvnet_pose_utils.project(kpt_3d, K, pose_pred)
        if cfg.smooth.enable:
            self.hist_pts.append(kpt_2d)
            pts_ = self.weighted_pts()
            pose_pred = pvnet_pose_utils.pnp(kpt_3d, pts_, K)
        # pose_pred[0, 3] += 0.01
        corner_3d = np.array(meta['corner_3d'])
        corner_2d_pred = pvnet_pose_utils.project(corner_3d, K, pose_pred)
        plt.figure(dpi=200)
        plt.axis('off')
        plt.tight_layout(pad=0)
        # if cfg.demo_crop.x1 > 0 or cfg.demo_crop.y1 > 0:
        #     demo_image_full = output['demo_image_full']
        #     plt.imshow(demo_image_full)
        #     K_full = K.copy()
        #     K_full[0, 2] += cfg.demo_crop.x1
        #     K_full[1, 2] += cfg.demo_crop.y1
        #     corner_2d_pred = pvnet_pose_utils.project(corner_3d, K_full, pose_pred)
        # else:
        inp2 = inp.clone()
        inp2[output['mask'][0].cpu().bool()] *= 0.5
        inp2[output['mask'][0].cpu().bool()] += 0.5
        plt.imshow(inp2)
        plt.gca().add_patch(
            patches.Polygon(xy=corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b'))
        plt.gca().add_patch(
            patches.Polygon(xy=corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='b'))
        # plt.scatter(output['kpt_2d'][0].cpu()[:, 0], output['kpt_2d'][0].cpu()[:, 1], s=1)
        # kpt2d_vis = output['kpt_2d'][0].cpu()
        # for j, k in enumerate(kpt2d_vis):
        #     plt.annotate(str(j), (k[0], k[1]))
        if cfg.dbg:
            from easyhec.utils.vis3d_ext import Vis3D
            vis3d = Vis3D(
                xyz_pattern=('x', '-y', '-z'),
                out_folder="dbg",
                sequence="pvnet_forward",
                auto_increase=True,
                # enable=,
            )
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            vis3d.add_image(Image.open(img_buf), name='pred_box')
            img_buf.close()
            plt.close("all")

            vis3d.add_image(inp, name='inp')
            vis3d.add_image(output['mask'][0].cpu(), name='pred_mask')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.imshow(inp)
            plt.scatter(output['kpt_2d'][0].cpu()[:, 0], output['kpt_2d'][0].cpu()[:, 1], s=1, color='red')
            kpt2d_vis = output['kpt_2d'][0].cpu()
            for j, k in enumerate(kpt2d_vis):
                plt.annotate(str(j), (k[0], k[1]), )
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            vis3d.add_image(Image.open(img_buf), name='pred_kpt2d')
            img_buf.close()

            if depth is not None:
                fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                pts = depth_to_rect(fu, fv, cu, cv, depth)
                vis3d.add_point_cloud(pts, colors=inp.reshape(-1, 3),
                                      remove_plane=False,
                                      max_z=1.5)
                vis3d.add_mesh(transform_points(self.model['pts'] / 1000.0, matrix_3x4_to_4x4(pose_pred)),
                               self.model['faces'], name='pose_pred')
            print()
        # else:
        #     plt.savefig(osp.join(vis_dir, f"{output['global_step']:06d}.png"))
        # plt.close("all")
        return pose_pred

    def weighted_pts(self):
        weight_num = self.smooth_num
        std_inv = self.smooth_std
        pts_list = self.hist_pts
        weights = np.exp(-(np.arange(weight_num) / std_inv) ** 2)[::-1]  # wn
        pose_num = len(pts_list)
        if pose_num < weight_num:
            weights = weights[-pose_num:]
        else:
            pts_list = pts_list[-weight_num:]
        pts = np.sum(np.asarray(pts_list) * weights[:, None, None], 0) / np.sum(weights)
        return pts

    def load_depth(self, depth_path: str):
        if depth_path.endswith(".zarr"):
            depth = zarr.load(depth_path) * cfg.demo_depth_scale
        elif depth_path.endswith(".png"):
            depth = cv2.imread(depth_path, 2).astype(np.float32) / 1000.0 * cfg.demo_depth_scale
        else:
            raise NotImplementedError()
        depth = depth[cfg.demo_crop.y1:cfg.demo_crop.y2, cfg.demo_crop.x1:cfg.demo_crop.x2]
        return depth

    @property
    def coco(self):
        if self._coco is None:
            args = DatasetCatalog.get(cfg.test.dataset)
            self.ann_file = args['ann_file']
            self.coco = coco.COCO(self.ann_file)
        return self._coco
