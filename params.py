import os

# UNet parameters
depth = 3
start_filters = 64
dropout_prob = 0.25

# Model hyperparameters
learning_rate = 0.0005
num_epochs = 20
batch_size = 16
threshold = 0.75
img_height = 128
mean = 0.17347709834575653
std = 0.2102048248052597

# Dataset parameters from environment variables
user = os.getenv('USER')
dt_name = os.getenv('DT_NAME')
version = os.getenv('VERSION')
label_set = os.getenv('LABEL_SET')
filter = os.getenv('FILTER').split()
n_class = int(os.getenv('N_CLASS'))

# Directories from environment variables
input_segmentation = os.getenv('INPUT_SEGMENTATION')
output_segmentation = os.getenv('OUTPUT_SEGMENTATION')
output_model_train = os.getenv('OUTPUT_MODEL_TRAIN')
model_directory = os.getenv('MODEL_DIRECTORY')

# us_reconstruction
poses_filename = os.getenv('POSES_FILENAME')
img_folder_path = os.getenv('IMG_FOLDER_PATH')
output_pointcloud_dir = os.getenv('OUTPUT_POINTCLOUD_DIR')
imgs_height_cm = float(os.getenv('IMGS_HEIGHT_CM'))

# plot_pointcloud
pointcloud_filenames = os.getenv('POINTCLOUD_FILENAMES').split()

def get_colors(n):
    colors = [
        [255, 255, 0],  # Yellow
        [0, 0, 255],    # Blue
        [0, 255, 0],    # Green
        [255, 0, 0]     # Red
    ]
    return colors[0:n]
