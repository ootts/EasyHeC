import glob
import os
import os.path as osp

import loguru
import numpy as np
import sapien.core as sapien
import torch
import torch.nn as nn
import tqdm
from pymp import Planner

from easyhec.modeling.models.rb_solve.collision_checker import CollisionChecker
from easyhec.modeling.models.rb_solve.workspace_boundary import get_workspace_boundary
from easyhec.structures.sapien_kin import SAPIENKinematicsModelStandalone
from easyhec.utils import utils_3d, render_api
from easyhec.utils.pn_utils import random_choice, to_array
from easyhec.utils.vis3d_ext import Vis3D


class SpaceExplorer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.total_cfg = cfg
        self.cfg = cfg.model.space_explorer

        self.dbg = self.total_cfg.dbg

        ckpt_dir = self.total_cfg.output_dir
        ckpt_paths = sorted(glob.glob(osp.join(ckpt_dir, "model*.pth")))
        ckpt_path = ckpt_paths[-1]
        loguru.logger.info(f"Auto detect ckpt_path {ckpt_path}")

        ckpt = torch.load(ckpt_path, "cpu")
        self.history_dof6 = ckpt['model']['history_ops']
        self.dummy = nn.Parameter(torch.zeros(1))
        self.sk = SAPIENKinematicsModelStandalone(self.cfg.urdf_path)
        if self.cfg.self_collision_check.enable:
            self.pymp_planner = Planner(self.cfg.urdf_path,
                                        user_joint_names=None,
                                        ee_link_name=self.cfg.move_group,
                                        srdf=self.cfg.srdf_path,
                                        )
        if self.cfg.collision_check.enable:
            self.planner = CollisionChecker(self.cfg)
        if self.cfg.max_dist_constraint.enable:
            self.max_dist_center = self.compute_max_dist_center()

    def forward(self, dps):
        vis3d = Vis3D(
            xyz_pattern=("x", "y", "z"),
            out_folder="dbg",
            sequence="space_explorer",
            auto_increase=True,
            enable=self.dbg,
        )
        to_zero = dps.get("to_zero", False)
        history_dof6 = self.history_dof6
        keep = ~(history_dof6 == 0).all(dim=1)
        history_dof6 = history_dof6[keep]
        history_dof6 = history_dof6[self.cfg.start:]
        engine = sapien.Engine()

        scene = engine.create_scene()
        scene.set_timestep(1 / 100.0)

        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        urdf_path = self.cfg.urdf_path
        # load as a kinematic articulation
        builder = loader.load_file_as_articulation_builder(urdf_path)
        robot = builder.build(fix_root_link=True)

        active_joints = robot.get_active_joints()
        for joint_idx, joint in enumerate(active_joints):
            joint.set_drive_property(stiffness=1e6, damping=1e5)
        if to_zero:
            qposes = np.zeros((1, robot.dof))
            loguru.logger.info("using zero qpos choices!")
        else:
            loguru.logger.info("using sampled qpos choices!")
            qposes = self.sample_qposes(self.cfg.sample_dof, robot.dof)
        width, height = self.cfg.width, self.cfg.height
        K = np.array(self.cfg.K)
        history_Tc_c2b = utils_3d.se3_exp_map(history_dof6).permute(0, 2, 1).numpy()
        history_Tc_c2b, _ = random_choice(history_Tc_c2b,
                                          size=self.cfg.sample,
                                          dim=0, replace=False)
        history_cam_poses = np.linalg.inv(history_Tc_c2b)
        variances = []
        n_self_collision = 0
        n_has_selected = 0
        n_exceed_max_dist = 0
        n_collision = 0
        plan_results = {}

        pts_base = get_workspace_boundary()
        self.planner.add_point_cloud(pts_base)
        for qpos_idx in tqdm.trange(qposes.shape[0]):
            vis3d.set_scene_id(qpos_idx)
            qpos = qposes[qpos_idx]
            qpos = [0] * self.cfg.qpos_choices_pad_left + qpos.tolist() + [0] * self.cfg.qpos_choices_pad_right
            robot.set_qpos(np.array(qpos, dtype=np.float32))
            vis3d.add_xarm(qpos)
            if self.cfg.self_collision_check.enable and self.total_cfg.use_xarm is True:
                self_collision = self.pymp_planner.robot.computeCollisions(qpos)
                if self_collision:
                    variances.append(0)
                    n_self_collision += 1
                    continue
            if self.cfg.max_dist_constraint.enable is True and self.total_cfg.use_xarm is True:
                vis3d.add_spheres(self.max_dist_center, radius=self.cfg.max_dist_constraint.max_dist)
                exceed_max_dist_constraint = False
                for link in range(len(self.sk.robot.get_links())):
                    pq = self.sk.compute_forward_kinematics(qpos, link)
                    if np.linalg.norm(pq.p - self.max_dist_center) > self.cfg.max_dist_constraint.max_dist:
                        exceed_max_dist_constraint = True
                        break
                if exceed_max_dist_constraint:
                    variances.append(0)
                    n_exceed_max_dist += 1
                    continue
            if self.cfg.collision_check.enable and self.total_cfg.use_xarm is True:
                curr_qpos = dps['qpos'][-1].cpu().numpy()
                pad_qpos = np.zeros([self.planner.robot.dof - curr_qpos.shape[0]])
                curr_qpos = np.concatenate([curr_qpos, pad_qpos])
                self.planner.robot.set_qpos(curr_qpos)
                timestep = self.cfg.collision_check.timestep
                code, result = self.planner.move_to_qpos(qpos, time_step=timestep, use_point_cloud=True,
                                                         planning_time=self.cfg.collision_check.planning_time)
                if code != 0:
                    n_collision += 1
                    vis3d.add_point_cloud(pts_base)
                    variances.append(0)
                    continue
                else:
                    plan_results[qpos_idx] = result
            elif self.total_cfg.use_xarm is False:
                rest_pose = np.array( ## Avoid achieving franka emika panda joint limits, so use fractioning
                                        [5.928617003472516e-05,
                                        -0.7848036409260933,
                                        -0.000308854746172659,
                                        -2.357726806912310,
                                        -0.00011798564528483742,
                                        1.570464383098814,
                                        0.7852387161304554,
                                        0.3,
                                        0.3]
                                    )
                target = (np.array(qpos) - rest_pose) / 2.3 + rest_pose
                plan_results[qpos_idx] = {"position":target.tolist()}
            masks = []
            for cam_pose in tqdm.tqdm(history_cam_poses, leave=False, disable="PYCHARM_HOSTED" in os.environ):
                rendered_mask = render_api.nvdiffrast_parallel_render_xarm_api(self.cfg.urdf_path,
                                                                               np.linalg.inv(cam_pose),
                                                                               qpos[:7] + [0, 0],
                                                                               height, width,
                                                                               to_array(K),
                                                                               robot_type = 0,
                                                                               return_ndarray=False)
                vis3d.add_image(rendered_mask)
                masks.append(rendered_mask)
            masks = torch.stack(masks)
            var = torch.var(masks.reshape(masks.shape[0], -1).float(), dim=0).sum()
            variances.append(var)
        variances = torch.tensor(variances)
        print(f"space exploring finished.")
        loguru.logger.info(f"total {qposes.shape[0]} qposes")
        loguru.logger.info(f"total {variances.shape[0]} variances")
        loguru.logger.info(f"total {n_has_selected} has selected")
        loguru.logger.info(f"total {n_self_collision} self_collision")
        loguru.logger.info(f"total {n_exceed_max_dist} exceed max dist constraint")
        loguru.logger.info(f"total {n_collision} collision")
        loguru.logger.info(f"valid qposes {(variances != 0).sum().item()}.\n")
        top_ids = variances.argsort(descending=True)
        vis3d.set_scene_id(0)
        rotx = utils_3d.rotx_np(-np.pi / 2)
        rotx = utils_3d.Rt_to_pose(rotx)
        tid = top_ids[0]
        if variances[tid] > 0:
            vis3d.add_xarm(qposes[tid], Tw_w2B=rotx, name=f'xarm')
            vis3d.increase_scene_id()
            next_qpos = qposes[tid]
            variance = variances[tid]
        else:
            raise RuntimeError(
                "no valid qpos found! Consider to increase the number of sampled qpos, or increase the max_dist.")

        outputs = {
            "qpos": next_qpos,
            "qpos_idx": tid,
            "variance": variance,
            "var_max": variances[variances > 0].max(),
            "var_min": variances[variances > 0].min(),
            "var_mean": variances[variances > 0].mean(),
            "plan_result": plan_results[tid.item()]
        }
        return outputs, {}

    def sample_qposes(self, dof, total_dof):
        joint_limits = [self.pymp_planner.robot.joint_limits[0][:dof],
                        self.pymp_planner.robot.joint_limits[1][:dof]]
        if 'PYCHARM_HOSTED' in os.environ and self.total_cfg.deterministic:
            np.random.seed(0)

        random_qpos = np.random.uniform(joint_limits[0], joint_limits[1], [self.cfg.n_sample_qposes, dof])
        pad_qpos = np.zeros([random_qpos.shape[0], total_dof - dof])
        random_qpos = np.concatenate([random_qpos, pad_qpos], axis=1)
        return random_qpos

    def compute_max_dist_center(self):
        pts = []
        qposes = np.random.uniform(*self.pymp_planner.robot.joint_limits,
                                   size=(self.cfg.max_dist_constraint.max_dist_center_compute_n,
                                         self.sk.robot.dof))
        loguru.logger.info("computing max dist center")
        for qpos in tqdm.tqdm(qposes):
            ret = self.pymp_planner.robot.computeCollisions(qpos)
            if not ret:
                curr_pts = []
                for link in range(len(self.sk.robot.get_links())):
                    pq = self.sk.compute_forward_kinematics(qpos, link)
                    curr_pts.append(pq.p)
                curr_pts = np.array(curr_pts)
                maxid = np.argmax(np.linalg.norm(curr_pts, axis=-1))
                pts.append(curr_pts[maxid])
            else:
                pts.append([0, 0, 0])
        pts = np.array(pts)
        maxi0 = pts[:, 1].argmax()
        mini0 = pts[:, 1].argmin()
        est_centerz = ((pts[maxi0] + pts[mini0]) / 2)[2]
        center = np.array([0, 0, est_centerz])
        loguru.logger.info("using center: " + str(center))
        return center
