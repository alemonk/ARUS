
# Dataset parameters
user = 'alemonk'
dt_name = 'phantom3'
version = 'v0.2'
label_set = 'ground-truth'
filter = ['labeled', 'reviewed']

# UNet parameters
n_class = 1
depth = 3
start_filters = 64
dropout_prob = 0.3

# Model hyperparameters
learning_rate = 0.0005
num_epochs = 20
batch_size = 16
threshold = 0.75
img_height = 128
mean = 0.17347709834575653
std = 0.2102048248052597

# Directories
input_segmentation = f'ds/test-{dt_name}'
output_segmentation = f'segm/results-test-{dt_name}'
output_model_test = f'segm/results-training-{dt_name}'
model_directory = f'segm/model-{n_class}class-{dt_name}.model'

# us_reconstruction
poses_filename = f'recon/img_pose.txt'
img_folder_path = f'{output_segmentation}/output_segmentation'
output_pointcloud_dir = f'recon/pointclouds/{dt_name}'
imgs_height_cm = 6.0

# plot_pointcloud
pointcloud_filenames = [f'{output_pointcloud_dir}/{i}.txt' for i in range(n_class)]

def get_colors(n):
    colors = [
        [255, 255, 0],  # Yellow
        [0, 0, 255],    # Blue
        [0, 255, 0],    # Green
        [255, 0, 0]     # Red
    ]
    return colors[0:n]

# sensor_to_image_transf = np.array([
#     [0.98501951, -0.09266497,  0.04806903, 205.29482151],
#     [-0.1614265, -0.99285119, -0.07972393,  13.92010423],
#     [0.06064688,  0.07523109, -0.99293638, -43.1846821],
#     [0, 0, 0, 1]
# ])
