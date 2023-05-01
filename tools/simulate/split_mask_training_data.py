import glob
import os
import os.path as osp

import tqdm

from easyhec.engine.defaults import default_argument_parser, setup


def main():
    parser = default_argument_parser(default_config_file="configs/xarm7/simulate/mask_data.yaml")
    args = parser.parse_args()
    total_cfg = setup(args)
    cfg = total_cfg.sim_mask_data

    data_dir = cfg.outdir
    train_dir = osp.join(data_dir, "train")
    val_dir = osp.join(data_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    for split in [train_dir, val_dir]:
        os.makedirs(osp.join(split, "color"), exist_ok=True)
        os.makedirs(osp.join(split, "depth"), exist_ok=True)
        os.makedirs(osp.join(split, "gt_mask"), exist_ok=True)
        os.makedirs(osp.join(split, "qpos"), exist_ok=True)
        os.makedirs(osp.join(split, "campose"), exist_ok=True)
    n = len(glob.glob(osp.join(data_dir, "color", "*.png")))
    n_train = int(n * 0.8)
    color_files = sorted(glob.glob(osp.join(data_dir, "color", "*.png")))
    mask_files = sorted(glob.glob(osp.join(data_dir, "gt_mask", "*.png")))
    depth_files = sorted(glob.glob(osp.join(data_dir, "depth", "*.png")))
    qpos_files = sorted(glob.glob(osp.join(data_dir, "qpos", "*.txt")))
    campose_files = sorted(glob.glob(osp.join(data_dir, "campose", "*.txt")))
    for i in tqdm.tqdm(range(n_train)):
        dst_color_file = osp.join(train_dir, "color", "%06d.png" % i)
        dst_mask_file = osp.join(train_dir, "gt_mask", "%06d.png" % i)
        dst_depth_file = osp.join(train_dir, "depth", "%06d.png" % i)
        dst_qpos_file = osp.join(train_dir, "qpos", "%06d.txt" % i)
        dst_campose_file = osp.join(train_dir, "campose", "%06d.txt" % i)
        os.system("mv %s %s" % (color_files[i], dst_color_file))
        os.system("mv %s %s" % (mask_files[i], dst_mask_file))
        os.system("mv %s %s" % (depth_files[i], dst_depth_file))
        os.system("mv %s %s" % (qpos_files[i], dst_qpos_file))
        os.system("mv %s %s" % (campose_files[i], dst_campose_file))

    for i in tqdm.tqdm(range(n_train, n)):
        dst_color_file = osp.join(val_dir, "color", "%06d.png" % (i - n_train))
        dst_mask_file = osp.join(val_dir, "gt_mask", "%06d.png" % (i - n_train))
        dst_depth_file = osp.join(val_dir, "depth", "%06d.png" % (i - n_train))
        dst_qpos_file = osp.join(val_dir, "qpos", "%06d.txt" % (i - n_train))
        dst_campose_file = osp.join(val_dir, "campose", "%06d.txt" % (i - n_train))
        os.system("mv %s %s" % (color_files[i], dst_color_file))
        os.system("mv %s %s" % (mask_files[i], dst_mask_file))
        os.system("mv %s %s" % (depth_files[i], dst_depth_file))
        os.system("mv %s %s" % (qpos_files[i], dst_qpos_file))
        os.system("mv %s %s" % (campose_files[i], dst_campose_file))


if __name__ == '__main__':
    main()
