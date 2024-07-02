import numpy as np
import cv2 as cv
import os
import sys
import shutil
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from params import *

poses_filename = 'recon/img_pose.txt'
img_folder_path = 'segm/test_forearm_results/output_segmentation'
output_pointcloud_dir = 'recon/pointclouds'

store_full_image = False
shutil.rmtree(output_pointcloud_dir, ignore_errors=True)
time.sleep(1)
# sensor_to_image_transf = np.array([
#     [0.98501951, -0.09266497,  0.04806903, 205.29482151],
#     [-0.1614265, -0.99285119, -0.07972393,  13.92010423],
#     [0.06064688,  0.07523109, -0.99293638, -43.1846821],
#     [0, 0, 0, 1]
# ])
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
    rot_matrix = np.array([[1.0 - q[2, 2] - q[3, 3], q[1, 2] + q[3, 0], q[1, 3] - q[2, 0], trans[0]],
                           [q[1, 2] - q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] + q[1, 0], trans[1]],
                           [q[1, 3] + q[2, 0], q[2, 3] - q[1, 0], 1.0 - q[1, 1] - q[2, 2], trans[2]],
                           [0.0, 0.0, 0.0, 1.0]])
    return rot_matrix

def process_image(image_file, pose_line, sensor_to_image_transf, colors, store_full_image=False):
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

    offset_height = 0
    offset_width = 0
    scale_x = imgs_height_cm / image_height_pixels
    scale_y = scale_x  # Assuming uniform scaling

    cv_img = cv.imread(image_file)
    cv_img = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)

    for class_idx, color in enumerate(colors):
        mask = cv.inRange(cv_img, np.array(color), np.array(color))
        
        if class_idx == 0:
            print(f"Processing {image_file}")
        if not store_full_image:
            edges = cv.Canny(mask, 100, 200)
            height, width = edges.shape
        else:
            edges = mask
            height, width = mask.shape

        output_file_path = os.path.join(output_pointcloud_dir, f'{class_idx}.txt')
        os.makedirs(output_pointcloud_dir, exist_ok=True)
        
        borders = 0

        with open(output_file_path, 'a') as output_file:
            for w in range(borders, width - borders):
                for h in range(borders, height - borders):
                    if edges[h, w] == 255:  # Edge detected or mask value is white
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

# Read 6D poses
with open(poses_filename) as pose_file:
    pose_lines = pose_file.readlines()

# Sort image files by number index
image_files = os.listdir(img_folder_path)
image_indices = [int(file.split('.png')[0]) for file in image_files]
image_indices.sort()

sorted_image_files = [file for index in image_indices for file in image_files if str(index) == file.split('.png')[0]]

# Read the first image to determine the height in pixels
first_image_path = os.path.join(img_folder_path, sorted_image_files[0])
first_image = cv.imread(first_image_path)
image_height_pixels = first_image.shape[0]

colors = get_colors(n_class)

for image_file, pose_line in zip(sorted_image_files, pose_lines):
    full_image_path = os.path.join(img_folder_path, image_file)
    process_image(full_image_path, pose_line, sensor_to_image_transf, colors, store_full_image)
