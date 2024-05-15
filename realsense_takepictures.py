import pyrealsense2 as rs
import cv2
import numpy as np
import franka_interface

def capture_image(pose_id):
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    pipeline.start(config)
    
    # Skip 5 frames to give the Auto-Exposure time to adjust
    for _ in range(5):
        pipeline.wait_for_frames()

    # Get frameset of color
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    
    # Convert images to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())
    
    # Save the image
    cv2.imwrite(f'checkerboard_pose_{pose_id}.png', color_image)
    
    # Stop streaming
    pipeline.stop()
    
    return color_image

predefined_poses = [
    {"position": [0.4, 0.0, 0.3], "orientation": [1, 0, 0, 0]},  # Pose 1
    {"position": [0.45, 0.05, 0.3], "orientation": [1, 0, 0, 0]}, # Pose 2
    {"position": [0.35, -0.05, 0.3], "orientation": [1, 0, 0, 0]},# Pose 3
    {"position": [0.4, 0.1, 0.3], "orientation": [1, 0, 0, 0]},  # Pose 4
    {"position": [0.4, -0.1, 0.3], "orientation": [1, 0, 0, 0]}, # Pose 5
    {"position": [0.45, 0.0, 0.35], "orientation": [1, 0, 0, 0]},# Pose 6
    {"position": [0.35, 0.0, 0.25], "orientation": [1, 0, 0, 0]},# Pose 7
    {"position": [0.4, 0.05, 0.35], "orientation": [1, 0, 0, 0]},# Pose 8
    {"position": [0.4, -0.05, 0.25], "orientation": [1, 0, 0, 0]},# Pose 9
    {"position": [0.45, 0.1, 0.25], "orientation": [1, 0, 0, 0]},# Pose 10
    {"position": [0.35, -0.1, 0.35], "orientation": [1, 0, 0, 0]},# Pose 11
    {"position": [0.4, 0.0, 0.4], "orientation": [1, 0, 0, 0]}, # Pose 12
    {"position": [0.4, 0.05, 0.2], "orientation": [1, 0, 0, 0]},# Pose 13
    {"position": [0.4, -0.05, 0.4], "orientation": [1, 0, 0, 0]},# Pose 14
    {"position": [0.45, 0.0, 0.2], "orientation": [1, 0, 0, 0]}  # Pose 15
]

if __name__ == "__main__":
    robot = franka_interface.RobotInterface()
    pose_id = 0
    for pose in predefined_poses:
        robot.move_to(pose)
        capture_image(pose_id)
        pose_id += 1
