'''
This script applies a pre-trained U-Net model for image segmentation on a series of images.
It calculates the mean and standard deviation of the dataset for normalization,
defines transformations, and processes the images directly.
The script then applies the model to the test data, generating predicted masks and saving the results for further analysis.
It assumes a pre-trained U-Net model is available and the paths to the images and model are set appropriately.
The output is a set of segmented images.
This script is useful for tasks requiring image segmentation, such as medical imaging or object detection.
Modifications may be needed based on your specific setup and use case.
'''

import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from unet import UNet
import numpy as np
from helper_functions import resize_image, numerical_sort, get_colors

# Parameters
img_height = 128
n_class = 2
threshold = 0.9

# Calculate mean and std
image_dir = 'ds/test_forearm'

# Define transformations
mean = 0.17347709834575653
std = 0.2102048248052597
final_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

plt.switch_backend('Agg')

def load_and_transform_image(image_path, transform, img_height):
    image = Image.open(image_path).convert("L")
    image = resize_image(image, img_height)
    if transform:
        image = transform(image)
    return image

def run_unet_model(model_path, image_dir, transform, n_class, output_dir='segm/test_forearm_results'):
    # Load the saved best model
    model = UNet(n_class)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Define a list of colors for each class
    colors = get_colors()

    # List all image files in the directory
    image_files = sorted([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))], key=numerical_sort)

    with torch.no_grad():
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(image_dir, image_file)
            image = load_and_transform_image(image_path, transform, img_height).unsqueeze(0).float()  # Add batch dimension
            output = model(image)
            output = torch.sigmoid(output)  # Apply sigmoid to get predictions in range [0, 1]
            output = output[0].cpu().numpy()  # Remove batch dimension and move to CPU

            print(f'Analyzing {i}')

            # Save the combined predicted masks
            color_mask = np.zeros((output.shape[1], output.shape[2], 3), dtype=np.uint8)
            for k in range(output.shape[0]):  # Loop over each channel
                color_mask[output[k] > threshold] = colors[k % len(colors)]  # Use color corresponding to the class

            plt.imshow(color_mask)
            plt.axis('off')
            os.makedirs(f'{output_dir}/output_segmentation', exist_ok=True)
            plt.savefig(os.path.join(f'{output_dir}/output_segmentation', f'{i}.png'))
            plt.close()

            for k in range(output.shape[0]):  # Loop over each channel
                plt.imshow(output[k], cmap='gray', vmin=0, vmax=1)
                plt.axis('off')
                os.makedirs(output_dir, exist_ok=True)
                os.makedirs(f'{output_dir}/class{k}', exist_ok=True)
                plt.savefig(os.path.join(f'{output_dir}/class{k}', f'{i}.png'))
                plt.close()

run_unet_model('segm/best_model.model', image_dir, final_transform, n_class)
