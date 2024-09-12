import os.path as osp
import matplotlib.pyplot as plt
import imageio
import numpy as np
from easyhec.utils import render_api, utils_3d, plt_utils
from easyhec.utils.vis3d_ext import Vis3D

data_dir = "data/franka/offline_example"
qpos = np.loadtxt(osp.join(data_dir, 'qpos/000000.txt'))
K = np.loadtxt(osp.join(data_dir, 'K.txt'))
color = imageio.imread_v2(osp.join(data_dir, "color/000000.png"))
H, W, _ = color.shape
vis3d = Vis3D(
    xyz_pattern=('x', 'y', 'z'),
    out_folder="dbg",
    sequence="test_franka_init",
)
vis3d.add_franka(qpos)
#  you may need to tune these numbers manually.
Tb_b2c = utils_3d.calc_pose_from_lookat(np.radians([60]), np.radians([20]), 1, 1.3)[0]
vis3d.add_camera_trajectory(Tb_b2c[None])
#  you may need to tune these numbers manually.
Tb_b2c[0, 3] += 0.5
Tb_b2c[1, 3] += 0.5
Tb_b2c[2, 3] += 0.4
Tc_c2b = np.linalg.inv(Tb_b2c)
mask = render_api.nvdiffrast_render_franka_api("assets/franka/urdf/franka.urdf", Tc_c2b, qpos.tolist(), H, W, K)

# tune until the mask has reasonable overlap with the RGB image.
plt.imshow(plt_utils.vis_mask(color, mask))
plt.show()

print(np.array2string(Tc_c2b, separator=', '))
print('put this into the yaml file: model.rbsolver.init_Tc_c2b')
