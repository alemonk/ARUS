import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import re
import numpy as np
from params import *

class ImageMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dirs, transform=None):
        self.image_dir = image_dir
        self.mask_dirs = mask_dirs
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))], key=numerical_sort)
        self.n_classes = len(mask_dirs)
        self.img_height = img_height

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert("L")
        image = resize_image(image, self.img_height)
        
        # Load and combine masks from all mask directories
        masks = []
        for mask_dir in self.mask_dirs:
            mask_path = os.path.join(mask_dir, self.image_files[idx])
            mask_img = Image.open(mask_path).convert("L")
            mask_img = resize_image(mask_img, self.img_height)
            masks.append(np.array(mask_img) / 255.0)
        
        # Stack masks along the third dimension
        mask = np.stack(masks, axis=0)
        
        if self.transform:
            image = self.transform(image)
            mask = torch.from_numpy(mask)
            
        return image, mask

def resize_image(image, desired_height):
    width, height = image.size
    aspect_ratio = width / height
    new_height = desired_height
    new_width = int(desired_height * aspect_ratio)
    new_width = (new_width // 16) * 16
    new_height = (new_height // 16) * 16
    return image.resize((new_width, new_height), Image.LANCZOS)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        logits = F.softmax(logits, dim=1)
        logits = logits.contiguous().view(logits.size(0), logits.size(1), -1)
        targets = targets.contiguous().view(targets.size(0), targets.size(1), -1)
        intersection = (logits * targets).sum(dim=2)
        union = logits.sum(dim=2) + targets.sum(dim=2)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice = dice.mean(dim=1)

        return 1 - dice.mean()

# Calculate mean and std of dataset
def calculate_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    mean = 0.0
    std = 0.0
    for images, _ in loader:
        images = images.view(images.size(0), images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    return mean.item(), std.item()

# Helper function to sort filenames numerically
def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# Function to plot performance metrics and save the plot as an image file
def plot_performance(train_losses, val_losses, training_time, output_dir, filename='performance_plot.png'):
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss - Training time: {:.0f}m {:.0f}s'.format(training_time // 60, training_time % 60))
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# Function to calculate IoU, precision, and accuracy
def calculate_metrics(outputs, masks):
    with torch.no_grad():
        outputs = torch.sigmoid(outputs)  # Apply sigmoid to get predictions in range [0, 1]
        predicted = (outputs > 0.5).float()  # Threshold to get binary mask

        # Intersection over Union (IoU)
        intersection = (predicted * masks).sum()
        union = (predicted + masks).sum() - intersection
        iou = (intersection / (union + 1e-6)).item()  # Add small epsilon to avoid division by zero

        # Accuracy
        correct_pixels = torch.sum(torch.eq(predicted, masks).float())
        total_pixels = torch.numel(predicted)
        accuracy = (correct_pixels / total_pixels).item()

        return iou, accuracy
