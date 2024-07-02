import numpy as np
import open3d as o3d
import os
import time
from params import *

default_path = '/home/alekappe/catkin_ws/src/my_package/src/'
pointcloud_dir = f'{default_path}pointclouds'
colors = get_colors(n_class)

# Dictionary to track visibility of each class
class_visibility = {i: True for i in range(n_class)}
last_pressed = time.time()

def debounce(interval=1):
    global last_pressed
    current_time = time.time()
    if current_time - last_pressed > interval:
        last_pressed = current_time
        return True
    return False

def load_point_clouds(pointcloud_dir):
    pointclouds = []
    for idx, filename in enumerate(sorted(os.listdir(pointcloud_dir))):
        if filename.endswith('.txt'):
            pointcloud_filename = os.path.join(pointcloud_dir, filename)
            try:
                points = np.loadtxt(pointcloud_filename, delimiter=',', unpack=False)
                if points.ndim == 1:
                    points = np.expand_dims(points, axis=0)
                
                if points.shape[1] >= 3:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
                    color = np.array(colors[idx % len(colors)]) / 255.0
                    pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (points.shape[0], 1)))
                    pointclouds.append((pcd, filename))
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

# Initial load
pointclouds = load_point_clouds(pointcloud_dir)

# Visualization
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()

# Add the point clouds and coordinate axes
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
vis.add_geometry(coordinate_frame)

def add_geometries(vis, pointclouds):
    for i, (pcd, filename) in enumerate(pointclouds):
        vis.add_geometry(pcd)
        bbox = pcd.get_axis_aligned_bounding_box()
        class_idx = int(os.path.splitext(filename)[0])
        bbox_color = np.array(colors[class_idx % len(colors)]) / 255.0
        bbox.color = bbox_color
        vis.add_geometry(bbox)
        update_bounding_box_text(bbox, os.path.splitext(filename)[0])

def refresh_visualization(vis):
    view_control = vis.get_view_control()
    camera_params = view_control.convert_to_pinhole_camera_parameters()
    
    vis.clear_geometries()
    vis.add_geometry(coordinate_frame)
    filtered_pointclouds = [(pcd, filename) for i, (pcd, filename) in enumerate(pointclouds) if class_visibility[i]]
    add_geometries(vis, filtered_pointclouds)
    
    view_control.convert_from_pinhole_camera_parameters(camera_params)
    vis.poll_events()
    vis.update_renderer()

def update(vis):
    if not debounce():
        return
    print("-------------------------\nUpdating point cloud")
    global pointclouds
    pointclouds = load_point_clouds(pointcloud_dir)
    refresh_visualization(vis)

def toggle_visibility(vis, n):
    if not debounce():
        return
    print("-------------------------\nToggling class visibility")
    class_visibility[n] = not class_visibility[n]
    refresh_visualization(vis)

# Register key callbacks
vis.register_key_callback(ord(" "), lambda vis: update(vis))
for i in range(n_class):
    vis.register_key_callback(ord(str(i)), lambda vis, idx=i: toggle_visibility(vis, idx))

# Run the visualizer
time.sleep(1)
update(vis)
vis.run()
vis.destroy_window()
