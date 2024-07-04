import cv2
import os

def extract_frames(video_paths, output_folder, frame_interval=1):
    """
    Extract frames from multiple video files and save them sequentially in the output folder.
    
    :param video_paths: List of paths to the video files.
    :param output_folder: Folder where extracted frames will be saved.
    :param frame_interval: Interval for frame extraction. 1 means every frame, 2 means every other frame, etc.
    """
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    frame_count = 0
    extracted_count = 0
    
    for video_path in video_paths:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_filename = os.path.join(output_folder, f"{extracted_count}.jpg")
                cv2.imwrite(frame_filename, frame)
                extracted_count += 1
                print(f'extracted frame: {extracted_count}')
            
            frame_count += 1
        
        cap.release()
        print(f"Extracted frames from {video_path} \n")
    
    print(f"Extracted {extracted_count} frames in total and saved to {output_folder}")

# video_paths = ['dataset_videos/dataset-forearm-ventral.mov']
# output_folder = 'ds/dataset-forearm-ventral'
# frame_interval = 20

video_paths = ['dataset_videos/video-trimmed.mov']
output_folder = 'ds/test-forearm-ventral'
frame_interval = 1

extract_frames(video_paths, output_folder, frame_interval)
