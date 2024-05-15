import cv2
import numpy as np

def read_image(file_path):
    return cv2.imread(file_path)

def find_checkerboard(image, pattern_size=(7, 6)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    return ret, corners

def main():
    # Example list of file paths and corresponding robot poses
    # image_files = ['checkerboard_pose_0.png', 'checkerboard_pose_1.png', ...]
    # robot_poses = [pose_0, pose_1, ...]  # List of end-effector poses

    # Checkerboard pattern settings
    pattern_size = (7, 6)
    square_size = 0.025  # size of a square in meters

    # Prepare object points
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    object_points = []
    image_points = []

    for image_file in image_files:
        image = read_image(image_file)
        ret, corners = find_checkerboard(image, pattern_size)
        if ret:
            object_points.append(objp)
            image_points.append(corners)
    
    # Camera calibration (Intrinsic parameters)
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, gray.shape[::-1], None, None
    )

    # Solve hand-eye calibration
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    for i, pose in enumerate(robot_poses):
        # Get the rotation and translation vectors from the pose
        rvec, tvec = cv2.Rodrigues(pose[:3, :3])[0], pose[:3, 3]
        R_gripper2base.append(rvec)
        t_gripper2base.append(tvec)
        
        # Use solvePnP to find the checkerboard pose
        ret, rvec, tvec = cv2.solvePnP(
            object_points[i], image_points[i], camera_matrix, dist_coeffs
        )
        R_target2cam.append(rvec)
        t_target2cam.append(tvec)

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base, R_target2cam, t_target2cam
    )

    print("Hand-Eye Calibration Result:")
    print("Rotation matrix (Camera to Gripper):", R_cam2gripper)
    print("Translation vector (Camera to Gripper):", t_cam2gripper)

if __name__ == "__main__":
    main()
