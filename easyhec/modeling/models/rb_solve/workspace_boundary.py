import numpy as np
import trimesh

from easyhec.utils.pn_utils import random_choice
from easyhec.utils.vis3d_ext import Vis3D


def get_workspace_boundary():
    xmin, ymin, zmin, xmax, ymax, zmax = -0.2, -0.5, -0.5, 1.0, 0.5, 1.0
    box = trimesh.primitives.Box(extents=[xmax - xmin, ymax - ymin, zmax - zmin])
    box.apply_translation([(xmax + xmin) / 2, (ymax + ymin) / 2, (zmax + zmin) / 2])
    pts_ws = box.sample(20000)

    pts_plane = np.zeros([20000, 3])
    pts_plane[:, 0] = np.random.uniform(-0.2, 0.15, size=20000)
    pts_plane[:, 1] = np.random.uniform(-0.4, 0.4, size=20000)
    pts_plane[:, 2] = 0
    norm = np.linalg.norm(pts_plane, axis=1)
    keep = norm > 0.1
    pts_plane = pts_plane[keep]
    pts_base = np.concatenate([pts_ws, pts_plane], axis=0)

    pts_base, _ = random_choice(pts_base, 5000, dim=0, replace=False)
    return pts_base


def main():
    vis3d = Vis3D(
        xyz_pattern=('x', 'y', 'z'),
        out_folder="dbg",
        sequence="model_env",
    )
    pts_base = get_workspace_boundary()
    vis3d.add_point_cloud(pts_base)
    vis3d.add_xarm(np.zeros(9))


if __name__ == '__main__':
    main()
