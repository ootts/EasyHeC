import tqdm
import json

import numpy as np
import imageio
import glob
import os
import os.path as osp
import argparse

import trimesh.primitives

from easyhec.engine.defaults import default_argument_parser, setup
from easyhec.utils.utils_3d import calc_pts_diameter2

parser = default_argument_parser(default_config_file="configs/xarm7/simulate/pvnet_data.yaml")
args = parser.parse_args()
total_cfg = setup(args)
cfg = total_cfg.sim_pvnet_data

data_dir = cfg.outdir
raw_dir = osp.join(data_dir, "raw")
outdir = osp.join(data_dir, "pvnet_format")
os.makedirs(outdir, exist_ok=True)
for split in ['train', 'test']:
    os.makedirs(osp.join(outdir, split, "rgb"), exist_ok=True)
    os.makedirs(osp.join(outdir, split, "mask"), exist_ok=True)
    os.makedirs(osp.join(outdir, split, "pose"), exist_ok=True)

img_paths = sorted(glob.glob(osp.join(raw_dir, "color/*png")))

trainidx, testidx = 0, 0
K = np.loadtxt(osp.join(raw_dir, "K.txt"))

model = trimesh.load("assets/xarm7_zeropos.ply")
diameter = calc_pts_diameter2(model.sample(10000))
result = trimesh.exchange.ply.export_ply(model, encoding='ascii')
for split in ['train', 'test']:
    output_file = open(osp.join(outdir, split, 'model.ply'), "wb+")
    output_file.write(result)
    output_file.close()
    # camera
    np.savetxt(osp.join(outdir, split, 'camera.txt'), K)
    # diameter
    open(osp.join(outdir, split, 'diameter.txt'), 'w').write(f'{diameter}\n')
for i in tqdm.trange(len(img_paths)):
    if i % 10 == 0:
        split = 'test'
        relative_new_img_path = f"test/rgb/{testidx}.jpg"
        relative_new_mask_path = f"test/mask/{testidx}.png"
        relative_new_pose_path = f"test/pose/pose{testidx}.npy"
    else:
        split = 'train'
        relative_new_img_path = f"train/rgb/{trainidx}.jpg"
        relative_new_mask_path = f"train/mask/{trainidx}.png"
        relative_new_pose_path = f"train/pose/pose{trainidx}.npy"
    imgid = int(img_paths[i].split("/")[-1].rstrip(".png"))
    rgb = imageio.imread_v2(osp.join(raw_dir, f"color/{imgid:06d}.png"))
    mask = imageio.imread_v2(osp.join(raw_dir, f"gt_mask/{imgid:06d}.png"))[:, :, 0]
    pose = np.loadtxt(osp.join(raw_dir, f"Tc_c2b/{imgid:06d}.txt"))
    pose = pose[:3, :4]
    imageio.imsave(osp.join(outdir, relative_new_img_path), rgb)
    imageio.imsave(osp.join(outdir, relative_new_mask_path), mask[:, :, None])
    np.save(osp.join(outdir, relative_new_pose_path), pose)

    if split == 'train':
        trainidx += 1
    else:
        testidx += 1
