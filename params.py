
# UNet parameters
n_class = 3
depth = 3
start_filters = 64
dropout_prob = 0.3

# Model hyperparameters
learning_rate = 0.0001
num_epochs = 20
batch_size = 16
threshold = 0.75
img_height = 128
mean = 0.17347709834575653
std = 0.2102048248052597

# Directories
output_model_test = 'segm/test_results'
output_segmentation = 'segm/test_forearm_results'
model_save_path = 'segm/best_model.model'

# us_reconstruction
imgs_height_cm = 6.0

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
