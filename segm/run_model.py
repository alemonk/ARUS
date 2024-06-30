'''
This script applies a pre-trained U-Net model for image segmentation on a series of images.
It calculates the mean and standard deviation of the dataset for normalization,
defines transformations, and creates a custom dataset for handling the image and mask data.
The script then applies the model to the test data, generating predicted masks and saving the results for further analysis.
It assumes a pre-trained U-Net model is available and the paths to the images, masks, and model are set appropriately.
The output is a set of segmented images.
This script is useful for tasks requiring image segmentation, such as medical imaging or object detection.
Modifications may be needed based on your specific setup and use case.
'''

import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import re
from unet import UNet
from helper_functions import *

# Parameters
img_height = 128
batch_size = 16
n_class = 1
threshold = 0.3

# Calculate mean and std
image_dir = 'ds/test_forearm'
mask_dir = [image_dir] * n_class

# Define transformations
base_transform = transforms.ToTensor()

mean = 0.17347709834575653
std = 0.2102048248052597
# Define final transformations
final_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Define the test dataset and loader with final transformations
test_dataset = ImageMaskDataset(image_dir, mask_dir, transform=final_transform)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

def test_model(model_class, model_path, test_loader, n_class, output_dir='segm/test_forearm_results'):
    # Load the saved best model
    model = model_class(n_class)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Define a list of colors for each class
    colors = [[255, 255, 0], [0, 0, 255], [0, 255, 0], [255, 0, 0]]  # Add more colors if there are more classes

    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            images, masks = images.float(), masks.float()
            outputs = model(images)
            outputs = torch.sigmoid(outputs)  # Apply sigmoid to get predictions in range [0, 1]

            # Save the combined predicted masks
            for j in range(images.size(0)):
                print(f'Analyzing {i * test_loader.batch_size + j}')
                output = outputs[j].cpu().numpy()
                color_mask = np.zeros((output.shape[1], output.shape[2], 3), dtype=np.uint8)
                
                for k in range(output.shape[0]):  # Loop over each channel
                    color_mask[output[k] > threshold] = colors[k % len(colors)]  # Use color corresponding to the class

                plt.imshow(color_mask)
                plt.axis('off')
                os.makedirs(f'{output_dir}/output_segmentation', exist_ok=True)
                plt.savefig(os.path.join(f'{output_dir}/output_segmentation', f'{i * test_loader.batch_size + j}.png'))
                plt.close()
                
                for k in range(output.shape[0]):  # Loop over each channel
                    plt.imshow(output[k], cmap='gray', vmin=0, vmax=1)
                    plt.axis('off')
                    os.makedirs(output_dir, exist_ok=True)
                    os.makedirs(f'{output_dir}/class{k}', exist_ok=True)
                    plt.savefig(os.path.join(f'{output_dir}/class{k}', f'{i * test_loader.batch_size + j}.png'))
                    plt.close()

plt.switch_backend('Agg')

test_model(UNet, 'segm/best_model.model', test_loader, n_class)
