# pip install segments-ai
from segments import SegmentsClient, SegmentsDataset
from segments.utils import export_dataset
import os
import shutil
import time
import random
from PIL import Image
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from segments_key import key
from params import *

# Initialize a SegmentsDataset from the release file
client = SegmentsClient(key)
release = client.get_release(f'{user}/{dt_name}', version)
dataset = SegmentsDataset(release, labelset=label_set, filter_by=filter)

# Export to COCO panoptic segmentation format
export_dataset(dataset, export_format='coco-panoptic')

output_images = 'output/images'
output_masks = 'output/masks'

# Create directories for images and masks
os.makedirs(output_images, exist_ok=True)
os.makedirs(output_masks, exist_ok=True)
os.makedirs(f'{user}/{dt_name}/' + version, exist_ok=True)

# Path to the directory containing the images and labels
segments_dir = f'segments/{user}_{dt_name}/{version}'
print(segments_dir)

# Iterate over the files in the segments directory
for filename in os.listdir(segments_dir):
    print(filename)
    if filename.endswith('.jpg'):
        # Move image files to the images folder
        shutil.copy(os.path.join(segments_dir, filename), os.path.join(output_images, filename))
    elif filename.endswith('_coco-panoptic.png'):
        # Move label files to the masks folder
        shutil.copy(os.path.join(segments_dir, filename), os.path.join(output_masks, filename))

# Cleanup
shutil.rmtree('segments')
shutil.rmtree(user)
if os.path.exists(f'export_coco-panoptic_{user}_{dt_name}_{version}.json'):
    os.remove(f'export_coco-panoptic_{user}_{dt_name}_{version}.json')

print("Processing complete. Images and masks moved to their respective folders.")
shutil.rmtree(f'out/{dt_name}/train', ignore_errors=True)
shutil.rmtree(f'out/{dt_name}/validation', ignore_errors=True)
shutil.rmtree(f'out/{dt_name}/test', ignore_errors=True)

time.sleep(1)

# Constants
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
SHUFFLE_IMAGES = True

OUTPUT_IMAGES_DIR = 'output/images'
OUTPUT_MASKS_DIR = 'output/masks'
MASKS_FILENAME_SUFFIX = 'label_ground-truth_coco-panoptic'

CLASS_COLORS = {
    'class_0': (0, 113, 188),
    'class_1': (216, 82, 24),
    'class_2': (236, 176, 31)
}

# Ensure the output directories exist
assert os.path.exists(OUTPUT_IMAGES_DIR), "Output images directory does not exist."
assert os.path.exists(OUTPUT_MASKS_DIR), "Output masks directory does not exist."

# Remove any existing phantom_dataset directory
shutil.rmtree('phantom_dataset', ignore_errors=True)

# Get all image files
image_files = [f for f in os.listdir(OUTPUT_IMAGES_DIR) if f.endswith('.jpg')]

if SHUFFLE_IMAGES:
    random.shuffle(image_files)

# Calculate the number of images for each split
num_images = len(image_files)
num_train = int(num_images * TRAIN_RATIO)
num_val = int(num_images * VAL_RATIO)
num_test = num_images - num_train - num_val

# Split the image files
train_files = image_files[:num_train]
val_files = image_files[num_train:num_train + num_val]
test_files = image_files[num_train + num_val:]

def create_masks_for_colors(mask_path, split, file_index):
    mask_image = Image.open(mask_path).convert('RGB')
    mask_size = mask_image.size
    pixels = mask_image.load()
    
    # Create mask images for each class
    masks = {class_name: Image.new('L', mask_size) for class_name in CLASS_COLORS}
    mask_pixels = {class_name: mask.load() for class_name, mask in masks.items()}
    
    for y in range(mask_size[1]):
        for x in range(mask_size[0]):
            for class_name, color in CLASS_COLORS.items():
                if pixels[x, y] == color:
                    mask_pixels[class_name][x, y] = 255
    
    # Save the mask images
    for class_name, mask in list(masks.items())[:n_class]:
        if not os.path.exists(f'out/{dt_name}/{split}/masks_{class_name}'):
            os.makedirs(f'out/{dt_name}/{split}/masks_{class_name}', exist_ok=True)
        mask.save(os.path.join(f'out/{dt_name}/{split}/masks_{class_name}', f'{file_index}.png'))

def process_and_copy_files(file_list, split):
    for i, file in enumerate(file_list):
        base_filename = os.path.splitext(file)[0]
        image_src_path = os.path.join(OUTPUT_IMAGES_DIR, file)
        mask_src_path = os.path.join(OUTPUT_MASKS_DIR, f'{base_filename}_{MASKS_FILENAME_SUFFIX}.png')

        if os.path.exists(image_src_path) and os.path.exists(mask_src_path):
            if not os.path.exists(f'out/{dt_name}/{split}/images'):
                os.makedirs(f'out/{dt_name}/{split}/images', exist_ok=True)
            shutil.copy(image_src_path, os.path.join(f'out/{dt_name}/{split}/images', f'{i}.png'))
            create_masks_for_colors(mask_src_path, split, i)

        print(f'{split} organization: {round(100 * (i + 1) / len(file_list))} %')

# Process and copy the files to the respective directories
process_and_copy_files(train_files, 'train')
process_and_copy_files(val_files, 'validation')
process_and_copy_files(test_files, 'test')

print("Dataset split into train, validation, and test sets with renamed files and separate masks for each class.")
shutil.rmtree('output', ignore_errors=True)
