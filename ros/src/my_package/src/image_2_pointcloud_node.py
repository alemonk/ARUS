#!/usr/bin/env python3

import numpy as np
import cv2 as cv
import os
import rospy
import time

# Parameters
poses_filename = rospy.get_param('~poses_filename', 'recon/img_pose.txt')
img_folder_path = rospy.get_param('~img_folder_path', 'segm/test_forearm_results/output_segmentation')
output_pointcloud_filename = rospy.get_param('~output_pointcloud_filename', 'recon/pointcloud.txt')
imgs_height_cm = 6.0
poll_interval = 5  # Check every 5 seconds for new images

identity_matrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])
sensor_to_image_transf = identity_matrix

def quaternion_to_rotation_matrix(quat, trans):
    q = quat.copy()
    n = np.dot(q, q)
    if n < np.finfo(q.dtype).eps:
        return np.identity(4)
    q = q * np.sqrt(2.0 / n)
    q = np.outer(q, q)
    rot_matrix = np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] + q[3, 0], q[1, 3] - q[2, 0], trans[0]],
        [q[1, 2] - q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] + q[1, 0], trans[1]],
        [q[1, 3] + q[2, 0], q[2, 3] - q[1, 0], 1.0 - q[1, 1] - q[2, 2], trans[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])
    return rot_matrix

def process_image(image_file, pose_line, sensor_to_image_transf, output_pointcloud_filename, image_height_pixels):
    # Extract translation and quaternion from the pose line
    pose_data = pose_line.split(' ')
    trans_x = float(pose_data[0])
    trans_y = float(pose_data[1])
    trans_z = float(pose_data[2])
    quat_x = float(pose_data[3])
    quat_y = float(pose_data[4])
    quat_z = float(pose_data[5])
    quat_w = float(pose_data[6])

    quat = np.array([quat_w, quat_x, quat_y, quat_z])
    translation = np.array([trans_x, trans_y, trans_z])
    homogeneous_matrix = quaternion_to_rotation_matrix(quat, translation)

    scale_x = imgs_height_cm / image_height_pixels
    scale_y = scale_x  # Assuming uniform scaling

    cv_img = cv.imread(image_file)
    gray_img = cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY)
    
    # Apply Canny edge detector
    edges = cv.Canny(gray_img, 100, 200)
    
    height, width = edges.shape

    with open(output_pointcloud_filename, 'a') as output_file:
        for w in range(width):
            for h in range(height):
                if edges[h][w] == 255:  # Edge detected
                    sensor_x = scale_x * h
                    sensor_y = scale_y * w
                    
                    # Create a 4x1 vector for the sensor coordinates in homogeneous form
                    sensor_coords = np.array([sensor_x, sensor_y, 0, 1])
                    
                    # Apply the sensor_to_image_transf transformation
                    transformed_sensor_coords = np.dot(sensor_to_image_transf, sensor_coords)
                    
                    # Apply the homogeneous matrix to get the final 3D world coordinates
                    world_coords = np.dot(homogeneous_matrix, transformed_sensor_coords)
                    
                    # Write the 3D world coordinates to the output file
                    point_str = f"{world_coords[0]},{world_coords[1]},{world_coords[2]}"
                    output_file.write(point_str + '\n')

def main():
    rospy.init_node('pointcloud_generator_node')
    
    default_path = '/home/alekappe/catkin_ws/src/my_package/src/'
    poses_filename = rospy.get_param('~poses_filename', f'{default_path}img_pose.txt')
    img_folder_path = rospy.get_param('~img_folder_path', f'{default_path}test_forearm_results/output_segmentation')
    output_pointcloud_filename = rospy.get_param('~output_pointcloud_filename', f'{default_path}pointcloud.txt')

    if os.path.exists(output_pointcloud_filename):
        os.remove(output_pointcloud_filename)

    # read 6D poses
    with open(poses_filename) as pose_file:
        pose_lines = pose_file.readlines()

    # Read the first image to determine the height in pixels
    while not rospy.is_shutdown():
        image_files = sorted([f for f in os.listdir(img_folder_path) if f.endswith('.png')], key=lambda x: int(x.split('.png')[0]))
        
        if image_files:
            first_image_path = os.path.join(img_folder_path, image_files[0])
            first_image = cv.imread(first_image_path)
            image_height_pixels = first_image.shape[0]
            break
        else:
            rospy.loginfo('Waiting for images to be available...')
            rospy.sleep(poll_interval)

    # Process images in numerical order
    image_index = 0
    while not rospy.is_shutdown() and image_index < len(pose_lines):
        image_file = os.path.join(img_folder_path, f'{image_index}.png')
        if os.path.exists(image_file):
            process_image(image_file, pose_lines[image_index], sensor_to_image_transf, output_pointcloud_filename, image_height_pixels)
            os.remove(image_file)
            image_index += 1
        else:
            rospy.loginfo(f'Waiting for image {image_index}.png to be available...')
            rospy.sleep(poll_interval)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
