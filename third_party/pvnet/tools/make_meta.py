import numpy as np
import trimesh
import os.path as osp

from termcolor import colored

from handle_custom_dataset import read_ply_points
from lib.csrc.fps import fps_utils


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--K_path', default="")
    parser.add_argument('--model_path', default="")
    parser.add_argument('--outdir', default="")

    args = parser.parse_args()
    model_path = args.model_path
    print(colored(f"using {model_path}", "red"))
    mesh = trimesh.load_mesh(model_path)
    K = np.loadtxt(args.K_path)
    xM, yM, zM = mesh.vertices.max(0)
    xm, ym, zm = mesh.vertices.min(0)
    corners3d = np.array([[xm, ym, zm],
                          [xm, ym, zM],
                          [xm, yM, zm],
                          [xm, yM, zM],
                          [xM, ym, zm],
                          [xM, ym, zM],
                          [xM, yM, zm],
                          [xM, yM, zM]])
    center = corners3d.mean(0)
    ply_points = read_ply_points(model_path)
    fps = fps_utils.farthest_point_sampling(ply_points, 8, True)
    kpt3d = np.concatenate([fps, center[None]], axis=0)

    meta = {
        "corner_3d": corners3d,
        "kpt_3d": kpt3d,
        "K": K
    }

    np.save(osp.join(args.outdir, "meta.npy"), meta)


if __name__ == '__main__':
    main()
