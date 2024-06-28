import os
import shutil
import random
from PIL import Image

# Constants
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
SHUFFLE_IMAGES = True

OUTPUT_IMAGES_DIR = 'output/images'
OUTPUT_MASKS_DIR = 'output/masks'
MASKS_FILENAME_SUFFIX = 'label_ground-truth_coco-panoptic'

MUSCLE_LAYER_COLOR = (216, 82, 24)
BONE_COLOR = (0, 113, 188)

# Ensure the output directories exist
assert os.path.exists(OUTPUT_IMAGES_DIR), "Output images directory does not exist."
assert os.path.exists(OUTPUT_MASKS_DIR), "Output masks directory does not exist."

# Remove any existing phantom_dataset directory
shutil.rmtree('phantom_dataset', ignore_errors=True)

# Create directories for train, validation, and test sets
for split in ['train', 'validation', 'test']:
    os.makedirs(f'ds/{split}/images', exist_ok=True)
    os.makedirs(f'ds/{split}/masks_muscle_layer', exist_ok=True)
    os.makedirs(f'ds/{split}/masks_bone', exist_ok=True)

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

def create_masks_for_colors(mask_path, muscle_layer_save_path, bone_save_path):
    mask_image = Image.open(mask_path).convert('RGB')
    muscle_layer_mask = Image.new('L', mask_image.size)
    bone_mask = Image.new('L', mask_image.size)

    pixels = mask_image.load()
    muscle_layer_pixels = muscle_layer_mask.load()
    bone_pixels = bone_mask.load()

    for y in range(mask_image.size[1]):
        for x in range(mask_image.size[0]):
            if pixels[x, y] == MUSCLE_LAYER_COLOR:
                muscle_layer_pixels[x, y] = 255
            if pixels[x, y] == BONE_COLOR:
                bone_pixels[x, y] = 255

    muscle_layer_mask.save(muscle_layer_save_path)
    bone_mask.save(bone_save_path)

def process_and_copy_files(file_list, split):
    for i, file in enumerate(file_list):
        base_filename = os.path.splitext(file)[0]
        image_src_path = os.path.join(OUTPUT_IMAGES_DIR, file)
        mask_src_path = os.path.join(OUTPUT_MASKS_DIR, f'{base_filename}_{MASKS_FILENAME_SUFFIX}.png')

        new_image_name = f'{i}.png'
        new_muscle_layer_mask_name = f'{i}.png'
        new_bone_mask_name = f'{i}.png'

        if os.path.exists(image_src_path) and os.path.exists(mask_src_path):
            shutil.copy(image_src_path, os.path.join(f'ds/{split}/images', new_image_name))
            create_masks_for_colors(
                mask_src_path,
                os.path.join(f'ds/{split}/masks_muscle_layer', new_muscle_layer_mask_name),
                os.path.join(f'ds/{split}/masks_bone', new_bone_mask_name)
            )
        
        print(f'{split} ds organization: { round(100 * i / len(file_list)) } %')

# Process and copy the files to the respective directories
process_and_copy_files(train_files, 'train')
process_and_copy_files(val_files, 'validation')
process_and_copy_files(test_files, 'test')

print("Dataset split into train, validation, and test sets with renamed files and separate masks for muscle layer and bone.")
shutil.rmtree('output', ignore_errors=True)
