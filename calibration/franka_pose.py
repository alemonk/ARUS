import numpy as np
import franka_interface

def get_robot_pose():
    # Initialize the robot interface
    robot = franka_interface.RobotInterface()

    # Get the current joint angles
    joint_angles = robot.joint_angles()

    # Get the current end-effector pose
    end_effector_pose = robot.endpoint_pose()

    return joint_angles, end_effector_pose

if __name__ == "__main__":
    joint_angles, end_effector_pose = get_robot_pose()
    print("Joint Angles:", joint_angles)
    print("End Effector Pose:", end_effector_pose)
