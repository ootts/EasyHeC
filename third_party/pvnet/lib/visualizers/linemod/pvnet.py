import cv2
import torch
import zarr
from lib.datasets.dataset_catalog import DatasetCatalog
from lib.config import cfg
import pycocotools.coco as coco
import numpy as np
from lib.utils.pvnet import pvnet_config
import matplotlib.pyplot as plt
from lib.utils import img_utils
import matplotlib.patches as patches
from lib.utils.pvnet import pvnet_pose_utils
from lib.utils.vsd import inout
from lib.utils.icp import icp_utils

mean = pvnet_config.mean
std = pvnet_config.std


class Visualizer:

    def __init__(self):
        args = DatasetCatalog.get(cfg.test.dataset)
        self.ann_file = args['ann_file']
        self.coco = coco.COCO(self.ann_file)
        if cfg.test.icp:
            model = inout.load_ply(cfg.custom.model_path)
            self.model = model
            model['pts'] = model['pts'] * 1000.0
            width = cfg.custom.width
            height = cfg.custom.height
            self.icp_refiner = icp_utils.ICPRefiner(model, (width, height))

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

        _, ax = plt.subplots(1)
        ax.imshow(inp)
        ax.add_patch(
            patches.Polygon(xy=corner_2d_gt[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='red'))
        ax.add_patch(
            patches.Polygon(xy=corner_2d_gt[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='red'))
        ax.add_patch(
            patches.Polygon(xy=corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='green'))
        ax.add_patch(
            patches.Polygon(xy=corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='green'))
        plt.show()

    def icp_refine(self, pose_pred, depth_file, output, K):
        if depth_file.endswith(".zarr"):
            depth = zarr.load(depth_file) * 1000.0
        else:
            depth = cv2.imread(depth_file, 2).astype(np.float32)
        mask = torch.argmax(output['seg'], dim=1)[0].detach().cpu().numpy()
        if pose_pred[2, 3] <= 0:
            return pose_pred
        depth[mask != 1] = 0
        pose_pred_tmp = pose_pred.copy()
        pose_pred_tmp[:3, 3] = pose_pred_tmp[:3, 3] * 1000.0
        R_refined, t_refined = self.icp_refiner.refine(depth, pose_pred_tmp[:3, :3], pose_pred_tmp[:3, 3], K.copy(),
                                                       depth_only=True, max_mean_dist_factor=5.0)
        R_refined, _ = self.icp_refiner.refine(depth, R_refined, t_refined, K.copy(), no_depth=True)
        if cfg.dbg:
            from nds.utils.vis3d_ext import Vis3D
            from nds.utils.utils_3d import depth_to_rect, transform_points, matrix_3x4_to_4x4
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

    def visualize_demo(self, output, inp, meta, show=True, depth_path=None):
        inp = img_utils.unnormalize_img(inp[0], mean, std).permute(1, 2, 0)
        if output['kpts_2d_anno'] is None:
            kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()
            kpt_3d = np.array(meta['kpt_3d'])
        else:
            kpt_2d = output['kpts_2d_anno']
            kpt_3d = np.array(meta['kpt_3d'])[cfg.demo_use_anno_idx]

        
        K = np.array(meta['K'])

        pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)
        if cfg.test.icp:
            pose_pred = self.icp_refine(pose_pred.copy(), depth_path, output, K)
        corner_3d = np.array(meta['corner_3d'])
        corner_2d_pred = pvnet_pose_utils.project(corner_3d, K, pose_pred)

        _, ax = plt.subplots(1)
        ax.imshow(inp)
        ax.add_patch(
            patches.Polygon(xy=corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b'))
        ax.add_patch(
            patches.Polygon(xy=corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='b'))
        plt.show()

        return pose_pred

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
        plt.savefig('test.jpg')
        plt.close(0)
