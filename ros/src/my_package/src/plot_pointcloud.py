import numpy as np
import open3d as o3d
import os
from params import *

default_path = '/home/alekappe/catkin_ws/src/my_package/src/'
pointcloud_dir = f'{default_path}pointclouds'
colors = get_colors(n_class)

def load_point_clouds(pointcloud_dir):
    pointclouds = []
    for idx, filename in enumerate(sorted(os.listdir(pointcloud_dir))):
        if filename.endswith('.txt'):
            pointcloud_filename = os.path.join(pointcloud_dir, filename)
            try:
                points = np.loadtxt(pointcloud_filename, delimiter=',', unpack=False)
                if points.ndim == 1 and points.size == 0:
                    print(f"File {pointcloud_filename} is empty.")
                    continue
                elif points.ndim == 1:
                    points = np.expand_dims(points, axis=0)
                
                pcd = o3d.geometry.PointCloud()
                if points.shape[1] >= 3:
                    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
                    color = np.array(colors[idx % len(colors)]) / 255.0
                    pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (points.shape[0], 1)))
                    pointclouds.append(pcd)
                else:
                    print(f"Unexpected number of columns in the point cloud file: {points.shape[1]}")
            except Exception as e:
                print(f"Error loading point cloud {pointcloud_filename}: {e}")
    return pointclouds

def update_bounding_box_text(bbox, class_name):
    min_bound = bbox.get_min_bound()
    max_bound = bbox.get_max_bound()
    dimensions = max_bound - min_bound
    dimension_text = f"Class {class_name} Dimensions: {dimensions[0]:.2f} x {dimensions[1]:.2f} x {dimensions[2]:.2f} cm"
    print(dimension_text)
    return dimension_text

# Initial load
pointclouds = load_point_clouds(pointcloud_dir)

# Visualization
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()

# Add the point clouds
for pcd in pointclouds:
    vis.add_geometry(pcd)

# Add coordinate axes
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
vis.add_geometry(coordinate_frame)

def update(vis):
    pointclouds = load_point_clouds(pointcloud_dir)
    vis.clear_geometries()
    for pcd in pointclouds:
        vis.add_geometry(pcd)
    vis.add_geometry(coordinate_frame)
    
    print('-------------------------')
    print('Updating point clouds')

    # Update bounding box
    i = 0
    for pcd in pointclouds:
        bbox = pcd.get_axis_aligned_bounding_box()
        bbox_color = np.array(colors[i % len(colors)]) / 255.0  # Normalize the color values
        bbox.color = bbox_color
        
        vis.add_geometry(bbox)
        update_bounding_box_text(bbox, str(i))
        i = i+1

    vis.poll_events()
    vis.update_renderer()
    return False

# Add update callback
vis.register_key_callback(ord(" "), lambda vis: update(vis))  # Press space to update
update(vis)

# Run the visualizer
vis.run()

# Close the visualizer
vis.destroy_window()
