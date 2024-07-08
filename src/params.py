import os

def get_env_variable(name, default=None):
    value = os.getenv(name, default)
    if value is None:
        raise ValueError(f"Environment variable '{name}' is not set")
    return value

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
user = get_env_variable('USER')
dt_name = get_env_variable('DT_NAME')
version = get_env_variable('VERSION')
label_set = get_env_variable('LABEL_SET')
filter = get_env_variable('FILTER').split()
n_class = int(get_env_variable('N_CLASS'))

# Directories from environment variables
input_segmentation = get_env_variable('INPUT_SEGMENTATION')
output_segmentation = get_env_variable('OUTPUT_SEGMENTATION')
output_model_train = get_env_variable('OUTPUT_MODEL_TRAIN')
model_directory = get_env_variable('MODEL_DIRECTORY')

# us_reconstruction
poses_filename = get_env_variable('POSES_FILENAME')
img_folder_path = get_env_variable('IMG_FOLDER_PATH')
output_pointcloud_dir = get_env_variable('OUTPUT_POINTCLOUD_DIR')
imgs_height_cm = float(get_env_variable('IMGS_HEIGHT_CM'))

# plot_pointcloud
pointcloud_filenames = get_env_variable('POINTCLOUD_FILENAMES').split()

def get_colors(n):
    colors = [
        [255, 255, 0],  # Yellow
        [0, 0, 255],    # Blue
        [0, 255, 0],    # Green
        [255, 0, 0]     # Red
    ]
    return colors[0:n]
