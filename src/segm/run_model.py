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
import sys
import shutil
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import resize_image, numerical_sort
from params import *

# Define transformations
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

def run_unet_model(model_path, input_segmentation, transform, n_class, output_segmentation):
    # Load the saved best model
    model = UNet(n_class, depth, start_filters, dropout_prob)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Define a list of colors for each class
    colors = get_colors(n_class)

    # List all image files in the directory
    image_files = sorted([f for f in os.listdir(input_segmentation) if os.path.isfile(os.path.join(input_segmentation, f))], key=numerical_sort)

    with torch.no_grad():
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(input_segmentation, image_file)
            image = load_and_transform_image(image_path, transform, img_height).unsqueeze(0).float()  # Add batch dimension
            output = model(image)
            output = torch.sigmoid(output)  # Apply sigmoid to get predictions in range [0, 1]
            output = output[0].cpu().numpy()  # Remove batch dimension and move to CPU

            print(f'Analyzing {i}')

            # Save the combined predicted masks
            color_mask = np.zeros((output.shape[1], output.shape[2], 3), dtype=np.uint8)
            for k in range(output.shape[0]):  # Loop over each channel
                color_mask[output[k] > threshold] = colors[k % len(colors)]  # Use color corresponding to the class

            os.makedirs(f'{output_segmentation}/output_segmentation', exist_ok=True)
            Image.fromarray(color_mask).save(os.path.join(f'{output_segmentation}/output_segmentation', f'{i}.png'))

            for k in range(output.shape[0]):  # Loop over each channel
                gray_mask = (output[k] * 255).astype(np.uint8)
                os.makedirs(f'{output_segmentation}/class{k}', exist_ok=True)
                Image.fromarray(gray_mask).save(os.path.join(f'{output_segmentation}/class{k}', f'{i}.png'))

shutil.rmtree(output_segmentation, ignore_errors=True)
time.sleep(1)
run_unet_model(model_directory, input_segmentation, final_transform, n_class, output_segmentation)
