import pyrealsense2 as rs
import cv2
import numpy as np
import glob
import os

def capture_checkerboard_images(num_images=10, save_path="camera_calibration/"):
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    pipeline.start(config)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    image_count = 0
    print("Press 'c' to capture an image. Press 'q' to quit.")
    
    try:
        while image_count < num_images:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue
            
            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            
            # Display the image
            cv2.imshow('Checkerboard Image', color_image)
            
            # Wait for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):  # Press 'c' to capture the image
                cv2.imwrite(f"{save_path}/image-{image_count}.png", color_image)
                print(f"Captured image-{image_count}.png")
                image_count += 1
            elif key == ord('q'):  # Press 'q' to quit
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

def calibrate_camera(image_folder, checkerboard_size=(9, 6), square_size=0.025):
    # Termination criteria for corner sub-pixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points based on the checkerboard pattern
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    # Arrays to store object points and image points from all the images
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane
    
    # Get all images from the folder
    images = glob.glob(f'{image_folder}/*.png')
    
    for image_file in images:
        img = cv2.imread(image_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        
        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            # Draw and display the corners
            cv2.drawChessboardCorners(img, checkerboard_size, corners2, ret)
            cv2.imshow('Checkerboard', img)
            cv2.waitKey(500)
    
    cv2.destroyAllWindows()
    
    # Perform camera calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    if ret:
        print("Camera matrix:\n", camera_matrix)
        print("Distortion coefficients:\n", dist_coeffs)
    
    # Save the calibration results
    np.save("camera_calibration/camera_matrix.npy", camera_matrix)
    np.save("camera_calibration/dist_coeffs.npy", dist_coeffs)
    
    return camera_matrix, dist_coeffs

if __name__ == "__main__":
    # Step 1: Capture images using RealSense camera
    # capture_checkerboard_images(num_images=10, save_path="camera_calibration/")
    
    # Step 2: Calibrate camera using the captured images
    calibrate_camera(image_folder="camera_calibration", checkerboard_size=(9, 6), square_size=0.025)