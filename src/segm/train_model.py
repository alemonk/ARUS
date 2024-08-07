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
from tqdm import tqdm
import shutil
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import plot_performance, ImageMaskDataset, calculate_mean_std, DiceLoss
from params import *

# Calculate mean and std
dataset = ImageMaskDataset(f'out/{dt_name}/train/images', [f'out/{dt_name}/train/masks_class_0'], transform=transforms.ToTensor())
mean, std = calculate_mean_std(dataset)
print(f"Calculated mean: {mean}, std: {std}")

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

shutil.rmtree(output_model_train, ignore_errors=True)
if os.path.exists(model_directory):
    os.remove(model_directory)

# Training function with performance plotting, saving the best model, and early stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_save_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    train_losses = []
    val_losses = []

    start_time = time.time()
    
    for epoch in range(num_epochs):
        since = time.time()

        # Training phase
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, masks = images.float().to(device), masks.float().to(device)
            optimizer.zero_grad()
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.float().to(device), masks.float().to(device)
                outputs = model(images)
                outputs = torch.sigmoid(outputs)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f'Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Save the best model
        if val_loss < best_loss:
            print("*** Saving best model ***")
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, model_save_path)

        time_elapsed = time.time() - since
        print('Elapsed time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('___________________________________________\n')

    # Load the best model weights
    model.load_state_dict(best_model_wts)
    
    training_time = time.time() - start_time
    
    # Plot and save the training and validation loss
    plot_performance(train_losses, val_losses, training_time, output_model_train)

    return model

def test_model(model, model_path, test_loader, n_class, output_model_test):
    # Load the saved best model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    colors = get_colors(n_class)

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
                
                os.makedirs(f'{output_model_test}/comparison', exist_ok=True)
                plt.savefig(os.path.join(f'{output_model_test}/comparison', f'test_result_{i * test_loader.batch_size + j}.png'))
                plt.close(fig)

                output = outputs[j].cpu().numpy()
                color_mask = np.zeros((output.shape[1], output.shape[2], 3), dtype=np.uint8)
                
                for k in range(min(n_class, len(colors))):
                    color_mask[output[k] > threshold] = colors[k]

                plt.imshow(color_mask)
                plt.axis('off')
                os.makedirs(f'{output_model_test}/output_combined', exist_ok=True)
                plt.savefig(os.path.join(f'{output_model_test}/output_combined', f'{i * test_loader.batch_size + j}.png'))
                plt.close()

plt.switch_backend('Agg')

# Create datasets and dataloaders
if n_class == 1:
    # criterion = DiceLoss()
    criterion = nn.BCELoss()
else:
    # criterion = DiceLoss()
    criterion = nn.CrossEntropyLoss()

# Function to create dataset paths
def create_dataset_paths(base_dir, n_class):
    return [os.path.join(base_dir, f'masks_class_{i}') for i in range(n_class)]

# Create datasets and dataloaders
train_mask_paths = create_dataset_paths(f'out/{dt_name}/train', n_class)
train_dataset = ImageMaskDataset(f'out/{dt_name}/train/images', train_mask_paths, transform)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

val_mask_paths = create_dataset_paths(f'out/{dt_name}/validation', n_class)
val_dataset = ImageMaskDataset(f'out/{dt_name}/validation/images', val_mask_paths, transform)
val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

test_mask_paths = create_dataset_paths(f'out/{dt_name}/test', n_class)
test_dataset = ImageMaskDataset(f'out/{dt_name}/test/images', test_mask_paths, transform)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

# Initialize model
model = UNet(n_class, depth, start_filters, dropout_prob)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_directory)
test_model(model, model_directory, test_loader, n_class, output_model_train)
