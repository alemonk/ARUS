#!/usr/bin/env python3

import os
import rospy
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import re
import time
from unet import UNet

# Parameters
img_height = 128
batch_size = 16
poll_interval = 5  # Check every 5 seconds for new images

class ImageMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))], key=numerical_sort)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.image_files[idx])
        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        image = resize_keep_aspect(image, img_height)
        mask = resize_keep_aspect(mask, img_height)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = (mask > 0).float()
        return image, mask

def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def resize_keep_aspect(image, desired_height):
    width, height = image.size
    aspect_ratio = width / height
    new_height = desired_height
    new_width = int(desired_height * aspect_ratio)
    new_width = (new_width // 16) * 16
    new_height = (new_height // 16) * 16
    return image.resize((new_width, new_height), Image.LANCZOS)

def calculate_mean_std(loader):
    mean = 0.0
    std = 0.0
    for images, _ in loader:
        images = images.view(images.size(0), images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    return mean.item(), std.item()

def test_model(model_class, model_path, image_path, output_dir):
    # Instantiate model, loss function, and optimizer
    n_class = 1  # Assuming binary segmentation
    depth = 4
    start_filters = 64

    model = model_class(n_class, depth, start_filters)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/output_segmentation', exist_ok=True)
    os.makedirs(f'{output_dir}/comparison', exist_ok=True)

    base_transform = transforms.ToTensor()
    final_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load and preprocess image
    image = Image.open(image_path).convert("L")
    image = resize_keep_aspect(image, img_height)
    image_tensor = final_transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor.float())
        output = torch.sigmoid(output)
        output = (output > 0.5).float().squeeze().cpu().numpy()

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(image, cmap='gray')
        ax[0].set_title('Input Image')
        ax[0].axis('off')

        ax[1].imshow(output, cmap='gray')
        ax[1].set_title('Predicted Mask')
        ax[1].axis('off')

        result_index = os.path.basename(image_path).split('.')[0]
        plt.savefig(os.path.join(f'{output_dir}/comparison', f'test_result_{result_index}.png'))
        plt.close(fig)

        plt.imsave(os.path.join(f'{output_dir}/output_segmentation', f'{result_index}.png'), output, cmap='gray')
        rospy.loginfo(f'Segmentation complete on frame {result_index}')

def main():
    rospy.init_node('image_segmentation_node')
    default_path = '/home/alekappe/catkin_ws/src/my_package/src/'
    image_dir = rospy.get_param('~image_dir', f'{default_path}test_forearm')
    mask_dir = image_dir
    model_path = rospy.get_param('~model_path', f'{default_path}best_model.model')
    output_dir = rospy.get_param('~output_dir', f'{default_path}test_forearm_results')

    base_transform = transforms.ToTensor()
    temp_dataset = ImageMaskDataset(image_dir, mask_dir, transform=base_transform)
    temp_loader = DataLoader(temp_dataset, batch_size=1, shuffle=False, num_workers=0)
    global mean, std
    mean, std = calculate_mean_std(temp_loader)
    rospy.loginfo(f"Calculated mean: {mean}, std: {std}")

    while not rospy.is_shutdown():
        image_files = sorted([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))], key=numerical_sort)
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            test_model(UNet, model_path, image_path, output_dir)
            os.remove(image_path)
        rospy.sleep(poll_interval)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
