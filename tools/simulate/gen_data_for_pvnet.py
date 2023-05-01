import os
import os.path as osp

import cv2
import imageio
import numpy as np
import sapien.core as sapien
import tqdm
import trimesh.primitives
from sapien.utils import Viewer

from easyhec.utils import render_api, plt_utils, utils_3d

from easyhec.engine.defaults import default_argument_parser, setup
from easyhec.utils.os_utils import number_of_monitors, red
from easyhec.utils.render_api import get_ring_object_poses
from easyhec.utils.utils_3d import rotx_np, roty_np
from easyhec.utils.vis3d_ext import Vis3D


def main():
    parser = default_argument_parser(default_config_file="configs/xarm7/simulate/pvnet_data.yaml")
    args = parser.parse_args()
    total_cfg = setup(args)
    cfg = total_cfg.sim_pvnet_data

    engine = sapien.Engine()

    sapien.render_config.camera_shader_dir = "rt"
    sapien.render_config.viewer_shader_dir = "rt"
    sapien.render_config.rt_samples_per_pixel = 16
    sapien.render_config.rt_use_denoiser = True

    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    scene = engine.create_scene()
    scene.set_timestep(1 / 100.0)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    urdf_path = cfg.urdf_path
    # load as a kinematic articulation
    builder = loader.load_file_as_articulation_builder(urdf_path)
    robot = builder.build(fix_root_link=True)

    assert robot, 'URDF not loaded.'

    if cfg.add_desk_cube.enable:
        actor_builder = scene.create_actor_builder()
        actor_builder.add_box_collision(half_size=cfg.add_desk_cube.half_size)
        actor_builder.add_box_visual(half_size=cfg.add_desk_cube.half_size,
                                     color=cfg.add_desk_cube.color)
        box = actor_builder.build(name='box')  # Add a box
        box.set_pose(sapien.Pose(p=cfg.add_desk_cube.pose))
    scene.set_ambient_light([1] * 3)
    scene.add_directional_light([0, 1, -1], [1.0, 1.0, 1.0], position=[0, 0, 2], shadow=True)

    # ---------------------------------------------------------------------------- #
    # Camera
    # ---------------------------------------------------------------------------- #
    near, far = 0.1, 100
    width, height = cfg.width, cfg.height
    camera_mount_actor = scene.create_actor_builder().build_kinematic()
    actor = camera_mount_actor
    camera = scene.add_mounted_camera(
        name="camera",
        actor=actor,
        pose=sapien.Pose(),
        width=width,
        height=height,
        fovy=np.deg2rad(135),
        near=near,
        far=far,
    )
    outdir = cfg.outdir
    outdir = osp.join(outdir, "raw")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(osp.join(outdir, "color"), exist_ok=True)
    os.makedirs(osp.join(outdir, "depth"), exist_ok=True)
    os.makedirs(osp.join(outdir, "gt_mask"), exist_ok=True)
    os.makedirs(osp.join(outdir, "Tc_c2b"), exist_ok=True)
    K = np.array(cfg.K)
    fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    camera.set_perspective_parameters(0.01, 100, fu, fv, cu, cv, 0.0)
    np.savetxt(osp.join(outdir, "K.txt"), K)

    active_joints = robot.get_active_joints()
    for joint_idx, joint in enumerate(active_joints):
        joint.set_drive_property(stiffness=1e6, damping=1e4)
    object_poses = get_ring_object_poses(cfg.min_dist, cfg.max_dist, cfg.min_elev, cfg.max_elev,
                                         cfg.n_dist, cfg.n_elev, cfg.nazim, trans_noise=cfg.trans_noise)

    Tc_c2bs = object_poses.cpu().numpy()

    coord_convert = utils_3d.Rt_to_pose(roty_np(-np.pi / 2) @ rotx_np(np.pi / 2))
    vis3d = Vis3D(
        xyz_pattern=("x", "-y", "-z"),
        out_folder="dbg",
        sequence="gen_data_for_pvnet",
        auto_increase=True,
        enable=total_cfg.dbg
    )
    qpose = np.zeros(7)
    robot.set_qpos(np.array(qpose, dtype=np.float32))  # must have this line!
    index = 0
    active_joints = robot.get_active_joints()
    for joint_idx, joint in enumerate(active_joints):
        joint.set_drive_property(stiffness=1e6, damping=1e4)
    nmonitors = number_of_monitors()
    if total_cfg.dbg and nmonitors > 0:
        viewer = Viewer(renderer)
        viewer.set_scene(scene)
        viewer.set_camera_xyz(x=-2, y=0, z=1)
        viewer.set_camera_rpy(r=0, p=-0.3, y=0)
    for Tc_c2b in tqdm.tqdm(Tc_c2bs):
        envmap = np.random.choice(cfg.envmaps)
        scene.set_environment_map_from_files(
            f"assets/envmaps/{envmap}/posx.jpg",
            f"assets/envmaps/{envmap}/negx.jpg",
            f"assets/envmaps/{envmap}/posy.jpg",
            f"assets/envmaps/{envmap}/negy.jpg",
            f"assets/envmaps/{envmap}/posz.jpg",
            f"assets/envmaps/{envmap}/negz.jpg")

        lights = []
        pts = trimesh.primitives.Sphere(radius=2.0).sample(cfg.n_point_light)
        for pt in pts:
            if pt[2] > 0.5:
                light = scene.add_point_light(pt, [1, 1, 1], shadow=True)
                lights.append(light)
        camera_pose = np.linalg.inv(Tc_c2b)
        camera_mount_actor.set_pose(sapien.Pose.from_transformation_matrix(camera_pose @ coord_convert))
        scene.update_render()
        if total_cfg.dbg and nmonitors > 0:
            viewer.toggle_pause(True)
            while not viewer.closed:
                qf = robot.compute_passive_force(
                    gravity=True,
                    coriolis_and_centrifugal=True,
                )
                robot.set_qf(qf)
                scene.update_render()
                # scene.step()
                viewer.render()
        camera.take_picture()

        rgba = camera.get_color_rgba()  # [H, W, 4]
        rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
        img = rgba_img[:, :, :3]
        vis3d.add_image(img, name='img')

        rendered_mask = render_api.nvdiffrast_render_xarm_api(urdf_path, np.linalg.inv(camera_pose),
                                                              np.zeros(7), height, width, K)

        vis3d.add_image(rendered_mask, name='mask')
        imageio.imsave(osp.join(outdir, f"gt_mask/{index:06d}.png"),
                       np.repeat(rendered_mask[:, :, None], 3, axis=-1).astype(np.uint8) * 255)
        tmp = plt_utils.vis_mask(img, rendered_mask.astype(np.uint8), color=[255, 0, 0])
        vis3d.add_image(tmp, name="hover")
        imageio.imsave(osp.join(outdir, f"color/{index:06d}.png"), img)
        np.savetxt(osp.join(outdir, f"Tc_c2b/{index:06d}.txt"), Tc_c2b)

        position = camera.get_float_texture('Position')  # [H, W, 4]

        points_opengl = position[..., :3].reshape(-1, 3)
        points_color = rgba[..., :3].reshape(-1, 3)
        model_matrix = camera.get_model_matrix()
        points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3]
        vis3d.add_point_cloud(points_world, points_color, name='points_world')
        pts_cam = utils_3d.transform_points(points_world, Tc_c2b)
        pts_cam[points_world[:, 2] == 0] = 0
        vis3d.add_point_cloud(pts_cam, name='pts_cam')
        fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        depth = pts_cam[:, 2].reshape(height, width)
        pts_rect = utils_3d.depth_to_rect(fu, fv, cu, cv, depth)
        vis3d.add_point_cloud(pts_rect, points_color, name='bp_from_depth')
        depth_image = (depth * 1000.0).astype(np.uint16)
        cv2.imwrite(osp.join(outdir, f"depth/{index:06d}.png"), depth_image)
        vis3d.increase_scene_id()

        for light in lights:
            scene.remove_light(light)
        index += 1


if __name__ == '__main__':
    main()
