import glob
import os.path as osp

import imageio
import numpy as np
import torch
import tqdm

from easyhec.utils import utils_3d, render_api, plt_utils
from easyhec.utils.vis3d_ext import Vis3D


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default="models/xarm7/example")
    parser.add_argument("--data_dir", default="data/xarm7/example", help="data dir to validate on")
    args = parser.parse_args()
    ckpt_dir = args.ckpt_dir
    data_dir = args.data_dir
    # data_dir = "data/realsense/20230402_152420"

    ckpt_path = sorted(glob.glob(osp.join(ckpt_dir, "*pth")))[-1]
    print(f"using ckpt path {ckpt_path}")
    # load solved Tc_c2b
    ckpt = torch.load(ckpt_path, map_location="cpu")
    dof6 = ckpt['model']['dof']
    Tc_c2b = utils_3d.se3_exp_map(dof6[None]).permute(0, 2, 1)[0].cpu().numpy()
    np.set_printoptions(suppress=True, precision=3)
    print("Tc_c2b", np.array2string(Tc_c2b, separator=","))
    # load data
    rgb_paths = sorted(glob.glob(osp.join(data_dir, "color/*.png")))
    qpos_paths = sorted(glob.glob(osp.join(data_dir, "qpos/*.txt")))
    K = np.loadtxt(osp.join(data_dir, "K.txt"))
    vis3d = Vis3D(
        xyz_pattern=('x', 'y', 'z'),
        out_folder="dbg",
        sequence="validate"
    )
    for i in tqdm.trange(len(rgb_paths)):
        vis3d.set_scene_id(i)
        rgb = imageio.imread_v2(rgb_paths[i])[:, :, :3]
        H, W = rgb.shape[:2]
        qpos = np.loadtxt(qpos_paths[i])
        rendered_mask = render_api.nvdiffrast_render_xarm_api("assets/xarm7_textured.urdf", Tc_c2b, qpos, H, W, K)
        tmp = plt_utils.vis_mask(rgb.copy(), rendered_mask.astype(np.uint8), [255, 0, 0])
        vis3d.add_image(tmp, name='rendered_mask')


if __name__ == '__main__':
    main()
