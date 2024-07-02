import numpy as np

# UNet parameters
num_epochs = 50
batch_size = 16
n_class = 3
threshold = 0.75

# model
img_height = 128
mean = 0.17347709834575653
std = 0.2102048248052597

# us_reconstruction
imgs_height_cm = 6.0
store_full_image = False
identity_matrix = np.eye(4)
sensor_to_image_transf = identity_matrix
# sensor_to_image_transf = np.array([
#     [0.98501951, -0.09266497,  0.04806903, 205.29482151],
#     [-0.1614265, -0.99285119, -0.07972393,  13.92010423],
#     [0.06064688,  0.07523109, -0.99293638, -43.1846821],
#     [0, 0, 0, 1]
# ])

def get_colors(n):
    colors = [
        [0, 0, 255],    # Blue
        [0, 255, 0],    # Green
        [255, 0, 0],    # Red
        [255, 255, 0],  # Yellow
    ]
    return colors[0:n]

