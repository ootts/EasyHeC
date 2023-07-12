import numpy as np
import sapien.core as sapien
import mplib

class CollisionChecker:
    def __init__(self, cfg):

        urdf_path = cfg.urdf_path
        srdf_path = cfg.srdf_path
        move_group = cfg.move_group
        self.engine = sapien.Engine()

        self.scene = self.engine.create_scene()

        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.robot: sapien.Articulation = loader.load(urdf_path)
        self.robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))

        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]

        self.planner = mplib.Planner(
            urdf=urdf_path,
            srdf=srdf_path,
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group=move_group,
            joint_vel_limits=np.ones(7),
            joint_acc_limits=np.ones(7))

    def add_point_cloud(self, point_cloud):
        """
        :param point_cloud: (N, 3) numpy array in arm base frame
        """
        # import trimesh
        # box = trimesh.creation.box([0.1, 0.4, 0.2])
        # points, _ = trimesh.sample.sample_surface(box, 1000)
        # points += [0.55, 0, 0.1]
        self.planner.update_point_cloud(point_cloud)
        return

    def move_to_pose(self, pose):
        """
        :param pose: 7D numpy array in arm base frame
        """
        # result = self.planner.plan_screw(pose, self.robot.get_qpos(), time_step=1 / 250,
        #                                  use_point_cloud=True)
        # if result['status'] != "Success":
        result = self.planner.plan(pose, self.robot.get_qpos(), time_step=1 / 250,
                                   use_point_cloud=True)
        if result['status'] != "Success":
            # print(result['status'])
            return -1, None
        # self.follow_path(result)
        return 0, result

    def move_to_qpos(self, target_qpos, mask=[],
                     time_step=0.1,
                     rrt_range=0.1,
                     planning_time=1,
                     fix_joint_limits=True,
                     use_point_cloud=False,
                     use_attach=False,
                     verbose=False):
        target_qpos = np.array(target_qpos)
        current_qpos = self.robot.get_qpos()
        self.planner.planning_world.set_use_point_cloud(use_point_cloud)
        self.planner.planning_world.set_use_attach(use_attach)
        n = current_qpos.shape[0]
        if fix_joint_limits:
            for i in range(n):
                if current_qpos[i] < self.planner.joint_limits[i][0]:
                    current_qpos[i] = self.planner.joint_limits[i][0] + 1e-3
                if current_qpos[i] > self.planner.joint_limits[i][1]:
                    current_qpos[i] = self.planner.joint_limits[i][1] - 1e-3
        self.planner.robot.set_qpos(current_qpos, True)
        collisions = self.planner.planning_world.collide_full()
        if len(collisions) != 0:
            print("Invalid start state!")
            for collision in collisions:
                print("%s and %s collide!" % (collision.link_name1, collision.link_name2))

        idx = self.planner.move_group_joint_indices

        self.planner.robot.set_qpos(current_qpos, True)
        status, path = self.planner.planner.plan(
            current_qpos[idx],
            [target_qpos[idx]],
            range=rrt_range,
            verbose=verbose,
            time=planning_time,
        )
        if status == "Exact solution":
            times, pos, vel, acc, duration = self.planner.TOPP(path, time_step)
            return 0, {
                "status": "Success",
                "time": times,
                "position": pos,
                "velocity": vel,
                "acceleration": acc,
                "duration": duration,
            }
        else:
            return -1, None
