from __future__ import print_function

import sys
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from loguru import logger
import time


try:
    from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi

    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))


from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

## END_SUB_TUTORIAL


def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if the values in two lists are within a tolerance of each other.
    For Pose and PoseStamped inputs, the angle between the two quaternions is compared (the angle
    between the identical orientations q and -q is calculated correctly).
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        # Euclidean distance
        d = dist((x1, y1, z1), (x0, y0, z0))
        # phi = angle between orientations
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

    return True


class MoveGroupPythonInterfaceTutorial(object):
    """MoveGroupPythonInterfaceTutorial"""

    def __init__(self):
        super(MoveGroupPythonInterfaceTutorial, self).__init__()

        ## BEGIN_SUB_TUTORIAL setup
        ##
        ## First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("move_group_python_interface_tutorial", anonymous=True)

        ## Instantiate a `RobotCommander`_ object. Provides information such as the robot's
        ## kinematic model and the robot's current joint states
        robot = moveit_commander.RobotCommander()

        ## Instantiate a `PlanningSceneInterface`_ object.  This provides a remote interface
        ## for getting, setting, and updating the robot's internal understanding of the
        ## surrounding world:
        scene = moveit_commander.PlanningSceneInterface()

        ## Instantiate a `MoveGroupCommander`_ object.  This object is an interface
        ## to a planning group (group of joints).  In this tutorial the group is the primary
        ## arm joints in the Panda robot, so we set the group's name to "panda_arm".
        ## If you are using a different robot, change this value to the name of your robot
        ## arm planning group.
        ## This interface can be used to plan and execute motions:
        group_name = "panda_arm"
        move_group = moveit_commander.MoveGroupCommander(group_name)

        ## Create a `DisplayTrajectory`_ ROS publisher which is used to display
        ## trajectories in Rviz:
        display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )

        ## END_SUB_TUTORIAL

        ## BEGIN_SUB_TUTORIAL basic_info
        ##
        ## Getting Basic Information
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^
        # We can get the name of the reference frame for this robot:
        planning_frame = move_group.get_planning_frame()
        print("============ Planning frame: %s" % planning_frame)

        # We can also print the name of the end-effector link for this group:
        eef_link = move_group.get_end_effector_link()
        print("============ End effector link: %s" % eef_link)

        # We can get a list of all the groups in the robot:
        group_names = robot.get_group_names()
        print("============ Available Planning Groups:", robot.get_group_names())

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        print("============ Printing robot state")
        print(robot.get_current_state())
        print("")
        ## END_SUB_TUTORIAL

        # Misc variables
        self.box_name = ""
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names

    def go_to_joint_state(self):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group

        ## BEGIN_SUB_TUTORIAL plan_to_joint_state
        ##
        ## Planning to a Joint Goal
        ## ^^^^^^^^^^^^^^^^^^^^^^^^
        ## The Panda's zero configuration is at a `singularity <https://www.quora.com/Robotics-What-is-meant-by-kinematic-singularity>`_, so the first
        ## thing we want to do is move it to a slightly better configuration.
        ## We use the constant `tau = 2*pi <https://en.wikipedia.org/wiki/Turn_(angle)#Tau_proposals>`_ for convenience:
        # We get the joint values from the group and change some of the values:
        joint_goal = move_group.get_current_joint_values()
        # joint_goal[0] = 0.5371091601179357
        # joint_goal[1] = 0.23245964375118555
        # joint_goal[2] = -0.49787666471799213
        # joint_goal[3] = -2.140900099558125
        # joint_goal[4] = -0.0011900663346880013
        # joint_goal[5] = 2.1542172839147202
        # joint_goal[6] = 0.4944201525971459

        joint_goal[0] = 5.928617003472516e-05
        joint_goal[1] = -0.7848036409260933
        joint_goal[2] = -0.000308854746172659
        joint_goal[3] = -2.357726806912310
        joint_goal[4] = -0.00011798564528483742
        joint_goal[5] = 1.570464383098814
        joint_goal[6] = 0.7852387161304554
        # [5.928617003472516e-05, -0.7848036409260933, -0.000308854746172659, -2.3577268069123103, -0.00011798564528483742, 1.570464383098814, 0.7852387161304554]

        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        move_group.go(joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        move_group.stop()

        ## END_SUB_TUTORIAL

        # For testing:
        current_joints = move_group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)

  
    def go_to_rest_pose(self):
        """
        Set the robot to the rest pose
        """
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group
        joint_goal = [5.928617003472516e-05,
                    -0.7848036409260933,
                    -0.000308854746172659,
                    -2.357726806912310,
                    -0.00011798564528483742,
                    1.570464383098814,
                    0.7852387161304554]
        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        move_group.go(joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        move_group.stop()

        ## END_SUB_TUTORIAL

        # For testing:
        current_joints = move_group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)
    
    def set_servo_angle(self, servo_id = None, angle = 0, is_radian = True, wait = True):
        """
        Set the angle of the specific servo_id
        """
        #logger.info("The robot 's {servo_id} is moved to {angle}".format(servo_id = servo_id, angle = angle))
        move_group = self.move_group
        if servo_id is None:
            
            if not isinstance(angle, list):
                joint_goal = angle.tolist()[:7]
            else:
                joint_goal = angle[:7]
            assert(len(joint_goal) == 7 or (len(joint_goal) == 9))
            # The go command can be called with joint values, poses, or without any
            # parameters if you have already set the pose or joint target for the group
            move_group.go(joint_goal[:7], wait=True)

            # Calling ``stop()`` ensures that there is no residual movement
            move_group.stop()
        else:
            assert isinstance(angle, int) or isinstance(angle, float)
            move_group[servo_id] = angle
            # The go command can be called with joint values, poses, or without any
            # parameters if you have already set the pose or joint target for the group
            move_group.go(joint_goal, wait=True)

            # Calling ``stop()`` ensures that there is no residual movement
            move_group.stop()
            # For testing:
        current_joints = move_group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)
        
    def get_servo_angle(self, servo_id = None, is_radian = True):
        if servo_id is not None:
            assert servo_id >= 1 and servo_id <= 7
        if servo_id is None:
            return 0, self.move_group.get_current_joint_values()
            
        else:
            return 0, self.move_group.get_current_joint_values()[servo_id]


    def get_joint_states(self):
        move_group = self.move_group
        return  move_group.get_current_joint_values()

    def go_to_pose_goal(self, x = 0.54093030529644, y = 0.25, z = 0.67):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group

        ## BEGIN_SUB_TUTORIAL plan_to_pose
        ##
        ## Planning to a Pose Goal
        ## ^^^^^^^^^^^^^^^^^^^^^^^
        ## We can plan a motion for this group to a desired pose for the
        ## end-effector:
        pose_goal = geometry_msgs.msg.Pose()

        pose_goal.orientation.x = 0.63519
        pose_goal.orientation.y = 0.2641
        pose_goal.orientation.z = -0.27939
        pose_goal.orientation.w = 0.66987
        pose_goal.position.x = x
        pose_goal.position.y = y + 0.098345534 + 0.10 ## 0.098345534 is the offset: the dist between joint 8 and true end effector
        pose_goal.position.z = z 
        move_group.set_pose_target(pose_goal)

        ## Now, we call the planner to compute the plan and execute it.
        # `go()` returns a boolean indicating whether the planning and execution was successful.
        success = move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets().
        move_group.clear_pose_targets()

        ## END_SUB_TUTORIAL

        # For testing:
        # Note that since this section of code will not be included in the tutorials
        # we use the class variable rather than the copied state variable
        current_pose = self.move_group.get_current_pose().pose


        time.sleep(1)
        move_group = self.move_group
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation.x = 0.63519
        pose_goal.orientation.y = 0.2641
        pose_goal.orientation.z = -0.27939
        pose_goal.orientation.w = 0.66987
        pose_goal.position.x = x
        pose_goal.position.y = y + 0.098345534 ## 0.098345534 is the offset: the dist between joint 8 and true end effector
        pose_goal.position.z = z 
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()




        return all_close(pose_goal, current_pose, 0.01)
    

if __name__ == "__main__":
    import numpy as np
    demo = MoveGroupPythonInterfaceTutorial()
    demo.go_to_pose_goal()
    