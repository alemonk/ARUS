import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import re
from unet import UNet

# Parameters
img_height = 128
batch_size = 16

# Calculate mean and std of dataset
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

# Function to resize images while maintaining aspect ratio and ensuring dimensions are divisible by 16
def resize_keep_aspect(image, desired_height):
    width, height = image.size
    aspect_ratio = width / height
    new_height = desired_height
    new_width = int(desired_height * aspect_ratio)

    # Make width and height divisible by 16
    new_width = (new_width // 16) * 16
    new_height = (new_height // 16) * 16

    return image.resize((new_width, new_height), Image.LANCZOS)

# Helper function to sort filenames numerically
def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# Define custom dataset
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

        # Ensure mask is binary (0 or 1)
        mask = (mask > 0).float()

        return image, mask

# Calculate mean and std
image_dir = 'ds/test_forearm'
mask_dir = image_dir

# Define transformations
base_transform = transforms.ToTensor()

# Create a temporary dataset for mean/std calculation
temp_dataset = ImageMaskDataset(image_dir, mask_dir, transform=base_transform)
temp_loader = DataLoader(temp_dataset, batch_size=1, shuffle=False, num_workers=0)
mean, std = calculate_mean_std(temp_loader)
print(f"Calculated mean: {mean}, std: {std}")

# Define final transformations
final_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Define the test dataset and loader with final transformations
test_dataset = ImageMaskDataset(image_dir, mask_dir, transform=final_transform)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

def test_model(model_class, model_path, test_loader, output_dir='segm/test_forearm_results'):
    # Load the saved best model
    model = model_class(n_class, depth, start_filters)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist
    os.makedirs(f'{output_dir}/output_segmentation', exist_ok=True)
    os.makedirs(f'{output_dir}/comparison', exist_ok=True)

    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            images, masks = images.float(), masks.float()
            outputs = model(images)
            outputs = torch.sigmoid(outputs)  # Apply sigmoid to get predictions in range [0, 1]
            outputs = (outputs > 0.5).float()  # Threshold to get binary mask

            for j in range(images.size(0)):
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                ax[0].imshow(images[j].cpu().numpy().squeeze(), cmap='gray')
                ax[0].set_title('Input Image')
                ax[0].axis('off')

                ax[1].imshow(outputs[j].cpu().numpy().squeeze(), cmap='gray')
                ax[1].set_title('Predicted Mask')
                ax[1].axis('off')

                plt.savefig(os.path.join(f'{output_dir}/comparison', f'test_result_{i * test_loader.batch_size + j}.png'))
                plt.close(fig)

            for j in range(images.size(0)):
                # Save only the output masks
                plt.imsave(os.path.join(f'{output_dir}/output_segmentation', f'{i * test_loader.batch_size + j}.png'), outputs[j].cpu().numpy().squeeze(), cmap='gray')
                print(f'Segmentation complete on frame {i * test_loader.batch_size + j}')

plt.switch_backend('Agg')

# Instantiate model, loss function, and optimizer
n_class = 1  # Assuming binary segmentation
depth = 4
start_filters = 64

test_model(UNet, 'segm/best_model.model', test_loader)
