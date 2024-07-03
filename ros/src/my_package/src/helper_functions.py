import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import re
import numpy as np

class ImageMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dirs, transform=None):
        self.image_dir = image_dir
        self.mask_dirs = mask_dirs
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))], key=numerical_sort)
        self.n_classes = len(mask_dirs)
        self.img_height = 128

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

# Helper function to sort filenames numerically
def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts
