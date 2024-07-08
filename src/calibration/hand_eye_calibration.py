import cv2
import numpy as np
import os
import glob

def get_robot_poses():
    # Generate random rotation matrix using random Euler angles
    random_euler = np.random.uniform(-np.pi, np.pi, size=(3,))
    random_rotation_matrix = cv2.Rodrigues(random_euler)[0]

    # Generate random translation vector within a reasonable range
    random_translation = np.random.uniform(1, 2, size=(3,))

    # Combine rotation and translation into a pose matrix
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = random_rotation_matrix
    pose_matrix[:3, 3] = random_translation

    return pose_matrix

def generate_pattern_points(square_size, pattern_size):
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size
    return pattern_points

def read_image(file_path):
    return cv2.imread(file_path)

def find_checkerboard(image, pattern_size=(9, 6)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if ret:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                   criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    return ret, corners

def main():
    # Example list of file paths and corresponding robot poses
    image_files = glob.glob('camera_calibration/*.png')
    robot_poses = [get_robot_poses() for _ in range(len(image_files))]

    # Load the camera matrix and distortion coefficients from the npy files
    camera_matrix = np.load('camera_calibration/camera_matrix.npy')
    dist_coeffs = np.load('camera_calibration/dist_coeffs.npy')

    # Solve hand-eye calibration
    R_endeffector2base = []
    t_endeffector2base = []
    R_target2cam = []
    t_target2cam = []

    for i, pose in enumerate(robot_poses):
        # Get the rotation and translation vectors from the pose
        R_endeffector2base.append(pose[:3, :3])
        t_endeffector2base.append(pose[:3, 3])

        # Read the image and find the checkerboard corners
        image = read_image(image_files[i])
        ret, corners = find_checkerboard(image)

        if ret:
            # Find the rotation and translation vectors.
            chessboard_coordinates = generate_pattern_points(square_size=0.025, pattern_size=(9, 6))
            ret, rvecs, tvecs = cv2.solvePnP(chessboard_coordinates, corners, camera_matrix, dist_coeffs)

            if ret:
                # Convert the rotation vector to a rotation matrix
                R, _ = cv2.Rodrigues(rvecs)
                R_target2cam.append(R)
                t_target2cam.append(tvecs)
            else:
                print(f"Failed to solve PnP for image {image_files[i]}")
        else:
            print(f"Checkerboard not found in image {image_files[i]}")

    if len(R_endeffector2base) > 0 and len(R_target2cam) > 0:
        R_cam2endeffector, t_cam2endeffector = cv2.calibrateHandEye(
            R_endeffector2base, t_endeffector2base, R_target2cam, t_target2cam
        )

        print("Hand-Eye Calibration Result:")
        print("Rotation matrix (Camera to End effector):", R_cam2endeffector)
        print("Translation vector (Camera to End effector):", t_cam2endeffector)

        # Save the calibration results
        save_path = "robot_calibration/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(f"{save_path}/R_cam2endeffector.npy", R_cam2endeffector)
        np.save(f"{save_path}/t_cam2endeffector.npy", t_cam2endeffector)
    else:
        print("Insufficient data for hand-eye calibration.")

if __name__ == "__main__":
    main()