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
import numpy as np
from helper_functions import *

# Parameters
original_height = 1600
original_width = 1000

# Unet parameters
num_epochs = 30
batch_size = 16
learning_rate = 0.001
n_class = 2
threshold = 0.3

# # Calculate mean and std
# image_dir = 'ds/train/images'
# mask_dir = 'ds/train/masks_bone'
# dataset = ImageMaskDataset(image_dir, mask_dir, transform=transforms.ToTensor())
# mean, std = calculate_mean_std(dataset)
# print(f"Calculated mean: {mean}, std: {std}")

# Define transformations
mean = 0.17347709834575653
std = 0.2102048248052597
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Gradient clipping function
def clip_gradients(model, max_norm=1.0):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.clamp_(-max_norm, max_norm)

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
            outputs = torch.sigmoid(outputs)
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
                outputs = torch.sigmoid(outputs)
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

def test_model(model_class, model_path, test_loader, n_class, output_dir='segm/test_results'):
    # Load the saved best model
    model = model_class(n_class)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    colors = [
        [255, 255, 0],  # Yellow for class 0
        [0, 0, 255],    # Blue for class 1
        [0, 255, 0],    # Green for class 2
        [255, 0, 0]     # Red for class 3
    ]

    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            images, masks = images.float(), masks.float()
            outputs = model(images)
            outputs = torch.sigmoid(outputs)  # Apply sigmoid to get predictions in range [0, 1]
            binary_outputs = (outputs > threshold).float()

            for j in range(images.size(0)):
                fig, ax = plt.subplots(n_class, 3, figsize=(12, 4 * n_class))
                if n_class == 1:
                    ax = np.expand_dims(ax, axis=0)  # Make sure ax is always 2D

                for k in range(n_class):
                    if k == 0:  # Plot the input image only once in the first column
                        ax[k, 0].imshow(images[j].cpu().numpy().squeeze(), cmap='gray')
                        ax[k, 0].set_title('Input Image')
                        ax[k, 0].axis('off')
                    else:
                        ax[k, 0].axis('off')  # Hide the extra axes

                    ax[k, 1].imshow(masks[j, k].cpu().numpy().squeeze(), cmap='gray')
                    ax[k, 1].set_title(f'Ground Truth Mask {k+1}')
                    ax[k, 1].axis('off')

                    ax[k, 2].imshow(binary_outputs[j, k].cpu().numpy().squeeze(), cmap='gray')
                    ax[k, 2].set_title(f'Predicted Mask {k+1}')
                    ax[k, 2].axis('off')
                
                os.makedirs(f'{output_dir}/comparison', exist_ok=True)
                plt.savefig(os.path.join(f'{output_dir}/comparison', f'test_result_{i * test_loader.batch_size + j}.png'))
                plt.close(fig)

                output = outputs[j].cpu().numpy()
                color_mask = np.zeros((output.shape[1], output.shape[2], 3), dtype=np.uint8)
                
                for k in range(min(n_class, len(colors))):
                    color_mask[output[k] > threshold] = colors[k]

                plt.imshow(color_mask)
                plt.axis('off')
                os.makedirs(f'{output_dir}/output_combined', exist_ok=True)
                plt.savefig(os.path.join(f'{output_dir}/output_combined', f'{i * test_loader.batch_size + j}.png'))
                plt.close()

plt.switch_backend('Agg')

# Create datasets and dataloaders
if n_class == 1:
    train_dataset = ImageMaskDataset('ds/train/images', ['ds/train/masks_bone'], transform)
    val_dataset = ImageMaskDataset('ds/validation/images', ['ds/validation/masks_bone'], transform)
    test_dataset = ImageMaskDataset('ds/test/images', ['ds/test/masks_bone'], transform)
    # criterion = DiceLoss()
    criterion = nn.BCELoss()
if n_class >= 2:
    train_dataset = ImageMaskDataset('ds/train/images', ['ds/train/masks_bone', 'ds/train/masks_muscle_layer'], transform)
    val_dataset = ImageMaskDataset('ds/validation/images', ['ds/validation/masks_bone', 'ds/validation/masks_muscle_layer'], transform)
    test_dataset = ImageMaskDataset('ds/test/images', ['ds/test/masks_bone', 'ds/test/masks_muscle_layer'], transform)
    # criterion = DiceLoss()
    criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

# Initialize model
model = UNet(n_class)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train and test model
shutil.rmtree('segm/test_results', ignore_errors=True)
if os.path.exists('segm/best_model.model'):
    os.remove('segm/best_model.model')

model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)
test_model(UNet, 'segm/best_model.model', test_loader, n_class)
