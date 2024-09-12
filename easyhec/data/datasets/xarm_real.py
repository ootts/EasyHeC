import cv2
import os.path as osp
import glob

import imageio
import numpy as np
import torch
import transforms3d

from easyhec.structures.sapien_kin import SAPIENKinematicsModelStandalone
from easyhec.utils import utils_3d


class XarmRealDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, data_dir, transforms=None, ds_len=-1):
        self.total_cfg = cfg
        self.cfg = cfg.dataset.xarm_real
        self.data_dir = data_dir
        if ds_len < 0:
            ds_len = 1000000
        rgb_paths = sorted(glob.glob(f"{data_dir}/color/*.png"))[:ds_len]
        mask_paths = sorted(glob.glob(f"{data_dir}/mask/*.png"))[:ds_len]
        qpos_paths = sorted(glob.glob(f"{data_dir}/qpos/*.txt"))[:ds_len]

        sk = SAPIENKinematicsModelStandalone(self.cfg.urdf_path)
        self.images = []
        self.masks = []
        self.qpos = []
        self.link_poses = []
        self.nimgs = len(rgb_paths)
        for rgb_path in rgb_paths:
            rgb = np.array(imageio.imread_v2(rgb_path))[..., :3]
            self.images.append(rgb)
        for mask_path in mask_paths:
            mask = cv2.imread(mask_path, 2) > 0
            self.masks.append(mask)
        if len(self.masks) > 0:
            self.masks = np.stack(self.masks)
            self.masks = torch.from_numpy(self.masks).float()

        for qpos_path in qpos_paths:
            qpos = np.loadtxt(qpos_path)
            self.qpos.append(qpos)
            link_poses = []
            for link in self.cfg.use_links:
                pad = np.zeros(sk.robot.dof - qpos.shape[0])
                pq = sk.compute_forward_kinematics(np.concatenate([qpos, pad]), link)
                R = transforms3d.quaternions.quat2mat(pq.q)
                t = pq.p
                pose_eb = utils_3d.Rt_to_pose(R, t)
                link_poses.append(pose_eb)
            link_poses = np.stack(link_poses)
            self.link_poses.append(link_poses)
        self.link_poses = np.stack(self.link_poses)
        self.link_poses = torch.from_numpy(self.link_poses).float()
        self.K = np.loadtxt(f"{data_dir}/K.txt")
        self.K = torch.from_numpy(self.K).float()
        Tc_c2b_path = f"{data_dir}/Tc_c2b.txt"
        if osp.exists(Tc_c2b_path):
            self.Tc_c2b = np.loadtxt(Tc_c2b_path)
        else:
            self.Tc_c2b = np.eye(4)
        self.Tc_c2b = torch.from_numpy(self.Tc_c2b).float()

    def __len__(self):
        return self.nimgs

    def __getitem__(self, idx):
        rgb = self.images[idx]
        qpos = self.qpos[idx]
        K = self.K
        Tc_c2b = self.Tc_c2b
        link_poses = self.link_poses[idx]
        mask = self.masks[idx]
        data_dict = {
            "rgb": rgb,
            "qpos": qpos,
            "K": K,
            "link_poses": link_poses,
            "Tc_c2b": Tc_c2b,
            "mask": mask
        }
        return data_dict
