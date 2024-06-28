import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from unet import UNet
import time
import copy
import re
import shutil
print('start')

# Parameters
num_epochs = 50
img_height = 128
batch_size = 16
learning_rate = 0.0001

# Dice loss function
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Apply sigmoid to get predictions in range [0, 1] if not already applied in the model
        inputs = torch.sigmoid(inputs)
        
        # Flatten label and prediction tensors if needed (this assumes batch processing)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

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
image_dir = 'ds/train/images'
mask_dir = 'ds/train/masks_bone'
dataset = ImageMaskDataset(image_dir, mask_dir, transform=transforms.ToTensor())
mean, std = calculate_mean_std(dataset)
print(f"Calculated mean: {mean}, std: {std}")

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Gradient clipping function
def clip_gradients(model, max_norm=1.0):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.clamp_(-max_norm, max_norm)

# Function to plot performance metrics and save the plot as an image file
def plot_performance(train_losses, val_losses, training_time, output_dir='segm/test_results', filename='performance_plot.png'):
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

# Training function with performance plotting and saving the best model
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    train_losses = []
    val_losses = []

    start_time = time.time()
    
    for epoch in range(num_epochs):
        since = time.time()

        model.train()
        running_loss = 0.0
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.float(), masks.float()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            clip_gradients(model)
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.float(), masks.float()
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Save the best model
        if val_loss < best_loss:
            print("Saving best model")
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, 'segm/best_model.model')

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('___________________________________________\n')

    # Load the best model weights
    model.load_state_dict(best_model_wts)
    
    training_time = time.time() - start_time
    
    # Plot and save the training and validation loss
    plot_performance(train_losses, val_losses, training_time)

    return model

def test_model(model_class, model_path, test_loader, output_dir='segm/test_results'):
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
                fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                ax[0].imshow(images[j].cpu().numpy().squeeze(), cmap='gray')
                ax[0].set_title('Input Image')
                ax[0].axis('off')

                ax[1].imshow(masks[j].cpu().numpy().squeeze(), cmap='gray')
                ax[1].set_title('Ground Truth Mask')
                ax[1].axis('off')

                ax[2].imshow(outputs[j].cpu().numpy().squeeze(), cmap='gray')
                ax[2].set_title('Predicted Mask')
                ax[2].axis('off')

                plt.savefig(os.path.join(f'{output_dir}/comparison', f'test_result_{i * test_loader.batch_size + j}.png'))
                plt.close(fig)

    iou, accuracy = calculate_metrics(outputs, masks)
    print(f'IoU: {iou}')
    print(f'Accuracy: {accuracy}')

plt.switch_backend('Agg')

# Create datasets and dataloaders
train_dataset = ImageMaskDataset('ds/train/images', 'ds/train/masks_bone', transform)
val_dataset = ImageMaskDataset('ds/validation/images', 'ds/validation/masks_bone', transform)
test_dataset = ImageMaskDataset('ds/test/images', 'ds/test/masks_bone', transform)

train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

# Instantiate model, loss function, and optimizer
n_class = 1  # Assuming binary segmentation
depth = 4
start_filters = 64
model = UNet(n_class, depth, start_filters)
criterion = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train and test model
shutil.rmtree('segm/test_results', ignore_errors=True)
if os.path.exists('segm/best_model.model'):
    os.remove('segm/best_model.model')

model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)
test_model(UNet, 'segm/best_model.model', test_loader)
