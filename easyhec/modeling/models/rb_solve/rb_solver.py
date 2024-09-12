import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import trimesh

from easyhec.structures.nvdiffrast_renderer import NVDiffrastRenderer
from easyhec.utils import utils_3d
from easyhec.utils.utils_3d import se3_log_map, se3_exp_map
from easyhec.utils.vis3d_ext import Vis3D

from loguru import logger

class RBSolver(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.total_cfg = cfg
        self.cfg = cfg.model.rbsolver

        self.dbg = self.total_cfg.dbg
        mesh_paths = self.cfg.mesh_paths
        for link_idx, mesh_path in enumerate(mesh_paths):
            mesh = trimesh.load(osp.expanduser(mesh_path), force = 'mesh')
            vertices = torch.from_numpy(mesh.vertices).float()
            faces = torch.from_numpy(mesh.faces).int()
            self.register_buffer(f'vertices_{link_idx}', vertices)
            self.register_buffer(f'faces_{link_idx}', faces)
        self.nlinks = len(mesh_paths)
        # camera parameters
        init_Tc_c2b = self.cfg.init_Tc_c2b
        init_dof = se3_log_map(torch.as_tensor(init_Tc_c2b, dtype=torch.float32)[None].permute(0, 2, 1), eps=1e-5,
                               backend="opencv")[0]
        self.dof = nn.Parameter(init_dof, requires_grad=True)
        # setup renderer
        self.H, self.W = self.cfg.H, self.cfg.W
        self.renderer = NVDiffrastRenderer([self.H, self.W])

        self.register_buffer(f'history_ops', torch.zeros(10000, 6))

    def forward(self, dps):
        vis3d = Vis3D(
            xyz_pattern=("x", "-y", "-z"),
            out_folder="dbg",
            sequence="rbsolver_forward",
            auto_increase=True,
            enable=self.dbg,
        )
        assert dps['global_step'] == 0
        put_id = (self.history_ops == 0).all(dim=1).nonzero()[0, 0].item()
        self.history_ops[put_id] = self.dof.detach()
        Tc_c2b = se3_exp_map(self.dof[None]).permute(0, 2, 1)[0]
        losses = []
        all_frame_all_link_si = []
        masks_ref = dps['mask']
        link_poses = dps['link_poses']
        K = dps['K'][0]

        batch_size = masks_ref.shape[0]
        for bid in range(batch_size):
            all_link_si = []
            for link_idx in range(self.nlinks):
                Tc_c2l = Tc_c2b @ link_poses[bid, link_idx]
                verts, faces = getattr(self, f"vertices_{link_idx}"), getattr(self, f"faces_{link_idx}")
                vis3d.add_mesh(utils_3d.transform_points(verts, Tc_c2l), faces, name=f"link{link_idx}")
                si = self.renderer.render_mask(verts, faces, K=K, object_pose=Tc_c2l)
                all_link_si.append(si)
            all_link_si = torch.stack(all_link_si).sum(0).clamp(max=1)
            all_frame_all_link_si.append(all_link_si)
            loss = torch.sum((all_link_si - masks_ref[bid].float()) ** 2)
            losses.append(loss)
        loss = torch.stack(losses).mean()
        all_frame_all_link_si = torch.stack(all_frame_all_link_si)
        output = {"rendered_masks": all_frame_all_link_si,
                  "ref_masks": masks_ref,
                  "error_maps": (all_frame_all_link_si - masks_ref.float()).abs(),
                  }
        # metrics
        gt_Tc_c2b = dps['Tc_c2b'][0]
        if not torch.allclose(gt_Tc_c2b, torch.eye(4).to(gt_Tc_c2b.device)):
            gt_dof6 = utils_3d.se3_log_map(gt_Tc_c2b[None].permute(0, 2, 1), backend='opencv')[0]
            trans_err = ((gt_dof6[:3] - self.dof[:3]) * 100).abs()
            rot_err = (gt_dof6[3:] - self.dof[3:]).abs().max() / np.pi * 180

            metrics = {
                "err_x": trans_err[0],
                "err_y": trans_err[1],
                "err_z": trans_err[2],
                "err_trans": trans_err.norm(),
                "err_rot": rot_err
            }
            output["metrics"] = metrics
        tsfm = utils_3d.se3_exp_map(self.dof[None].detach().cpu()).permute(0, 2, 1)[0]
        output['tsfm'] = tsfm
        loss_dict = {"mask_loss": loss}
        return output, loss_dict
