import numpy as np
import sapien.core as sapien


class SAPIENKinematicsModelStandalone:

    def __init__(self, urdf_path):
        self.engine = sapien.Engine()

        self.scene = self.engine.create_scene()

        loader = self.scene.create_urdf_loader()
        # loader.scale = 10
        builder = loader.load_file_as_articulation_builder(urdf_path)

        self.robot: sapien.Articulation = builder.build(fix_root_link=True)

        self.robot.set_pose(sapien.Pose())

        self.robot.set_qpos(np.zeros(self.robot.dof))

        self.scene.step()

        self.model: sapien.PinocchioModel = self.robot.create_pinocchio_model()

    def compute_forward_kinematics(self, qpos, link_index):
        assert len(qpos) == self.robot.dof, f"qpos {len(qpos)} != {self.robot.dof}"
        self.model.compute_forward_kinematics(np.array(qpos))

        return self.model.get_link_pose(link_index)

    def release(self):
        self.scene = None

        self.engine = None


def main():
    sk = SAPIENKinematicsModelStandalone("data/xarm7.urdf")
    qpos = np.loadtxt("/home/linghao/PycharmProjects/cam_robot_calib/data/realsense/20230124_092547/pose_eb_000000.txt")
    # print(sk.compute_forward_kinematics(qpos))


if __name__ == '__main__':
    main()
