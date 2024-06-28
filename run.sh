
# DOWNLOAD DATASET FROM SEGMENTS.AI AND ORGANIZE IT INTO TRAIN, VALIDATION AND TEST
"/Users/alemonk/Library/Mobile Documents/com~apple~CloudDocs/MSc THESIS/segmentation/myenv/bin/python" "/Users/alemonk/Library/Mobile Documents/com~apple~CloudDocs/MSc THESIS/segmentation/ds/download_dataset.py"
"/Users/alemonk/Library/Mobile Documents/com~apple~CloudDocs/MSc THESIS/segmentation/myenv/bin/python" "/Users/alemonk/Library/Mobile Documents/com~apple~CloudDocs/MSc THESIS/segmentation/ds/organize_dataset.py"

# EXTRACT FRAMES FROM FOREARM TEST DATASET
"/Users/alemonk/Library/Mobile Documents/com~apple~CloudDocs/MSc THESIS/segmentation/myenv/bin/python" "/Users/alemonk/Library/Mobile Documents/com~apple~CloudDocs/MSc THESIS/segmentation/ds/extract_frames.py"

# RUN MODEL
"/Users/alemonk/Library/Mobile Documents/com~apple~CloudDocs/MSc THESIS/segmentation/myenv/bin/python" "/Users/alemonk/Library/Mobile Documents/com~apple~CloudDocs/MSc THESIS/segmentation/segm/run_model.py"

# 3D RECONSTRUCTION
"/Users/alemonk/Library/Mobile Documents/com~apple~CloudDocs/MSc THESIS/segmentation/myenv/bin/python" "/Users/alemonk/Library/Mobile Documents/com~apple~CloudDocs/MSc THESIS/segmentation/recon/us_reconstruction.py"
"/Users/alemonk/Library/Mobile Documents/com~apple~CloudDocs/MSc THESIS/segmentation/myenv/bin/python" "/Users/alemonk/Library/Mobile Documents/com~apple~CloudDocs/MSc THESIS/segmentation/recon/plot_pointcloud.py"
