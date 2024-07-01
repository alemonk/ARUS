#!/usr/bin/env python3

import numpy as np
import cv2 as cv
import os
import rospy
from cv_bridge import CvBridge
from my_package.msg import ImagePose
import matplotlib.pyplot as plt

def get_colors(n):
    colors = [
        [255, 255, 0],  # Yellow
        [0, 0, 255],    # Blue
        [0, 255, 0],    # Green
        [255, 0, 0]     # Red
    ]
    return colors[0:n]

# Parameters
n_class = 2
colors = get_colors(n_class)
default_path = '/home/alekappe/catkin_ws/src/my_package/src/'
output_pointcloud_dir = f'{default_path}pointclouds'
imgs_height_cm = 6.0
identity_matrix = np.eye(4)
sensor_to_image_transf = identity_matrix

def quaternion_to_rotation_matrix(quat, trans):
    q = quat.copy()
    n = np.dot(q, q)
    if n < np.finfo(q.dtype).eps:
        return np.eye(4)
    q = q * np.sqrt(2.0 / n)
    q = np.outer(q, q)
    rot_matrix = np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] + q[3, 0], q[1, 3] - q[2, 0], trans[0]],
        [q[1, 2] - q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] + q[1, 0], trans[1]],
        [q[1, 3] + q[2, 0], q[2, 3] - q[1, 0], 1.0 - q[1, 1] - q[2, 2], trans[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])
    return rot_matrix

def process_image(image, pose, sensor_to_image_transf, image_height_pixels, colors):
    trans_x = pose.position.x
    trans_y = pose.position.y
    trans_z = pose.position.z
    quat_x = pose.orientation.x
    quat_y = pose.orientation.y
    quat_z = pose.orientation.z
    quat_w = pose.orientation.w

    quat = np.array([quat_w, quat_x, quat_y, quat_z])
    translation = np.array([trans_x, trans_y, trans_z])
    homogeneous_matrix = quaternion_to_rotation_matrix(quat, translation)

    offset_height = 0
    offset_width = 0
    scale_x = imgs_height_cm / image_height_pixels
    scale_y = scale_x  # Assuming uniform scaling

    cv_img = image
    cv_img = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
    
    for class_idx, color in enumerate(colors):
        mask = cv.inRange(cv_img, np.array(color), np.array(color))
        edges = cv.Canny(mask, 100, 200)
        
        print(f"Processing image for class {class_idx}")
        height, width = edges.shape

        output_file_path = os.path.join(output_pointcloud_dir, f'class_{class_idx}.txt')
        os.makedirs(output_pointcloud_dir, exist_ok=True)
        
        with open(output_file_path, 'a') as output_file:
            for w in range(width):
                for h in range(height):
                    if edges[h][w] == 255:  # Edge detected
                        sensor_x = scale_x * (h - offset_height)
                        sensor_y = scale_y * (w - offset_width)
                        
                        # Create a 4x1 vector for the sensor coordinates in homogeneous form
                        sensor_coords = np.array([sensor_x, sensor_y, 0, 1])
                        
                        # Apply the sensor_to_image_transf transformation
                        transformed_sensor_coords = np.dot(sensor_to_image_transf, sensor_coords)
                        
                        # Apply the homogeneous matrix to get the final 3D world coordinates
                        world_coords = np.dot(homogeneous_matrix, transformed_sensor_coords)
                        
                        # Write the 3D world coordinates to the output file
                        point_str = f"{world_coords[0]},{world_coords[1]},{world_coords[2]}"
                        output_file.write(point_str + '\n')

class PointCloudGeneratorNode:
    def __init__(self):
        rospy.init_node('pointcloud_generator_node')

        self.output_pointcloud_dir = rospy.get_param('~output_pointcloud_dir', output_pointcloud_dir)
        self.processed_image_topic = rospy.get_param('~processed_image_topic', 'processed_image_topic')

        if not os.path.exists(self.output_pointcloud_dir):
            os.makedirs(self.output_pointcloud_dir)

        self.bridge = CvBridge()
        self.subscriber = rospy.Subscriber(self.processed_image_topic, ImagePose, self.callback)

    def callback(self, image_pose_msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(image_pose_msg.image, "bgr8")
            pose = image_pose_msg.pose
            title = image_pose_msg.title

            # Determine the height of the image in pixels
            image_height_pixels = cv_img.shape[0]

            # Process the image to generate the point cloud
            process_image(cv_img, pose, sensor_to_image_transf, image_height_pixels, colors)
            rospy.loginfo(f"Processed image {title}")

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

def main():
    try:
        node = PointCloudGeneratorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
