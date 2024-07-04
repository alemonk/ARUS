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

user = 'ale'
dt_name = 'forearm-ventral-final-clone'
version = 'v0.3'
label_set = 'ground-truth'
filter = ['labeled', 'reviewed']

dt_name_full = f'{user}/{dt_name}'

# Initialize a SegmentsDataset from the release file
client = SegmentsClient(key)

release = client.get_release(dt_name_full, version)
dataset = SegmentsDataset(release, labelset=label_set, filter_by=filter)

# Export to COCO panoptic segmentation format
export_dataset(dataset, export_format='coco-panoptic')

output_images = 'output/images'
output_masks = 'output/masks'

# Create directories for images and masks
os.makedirs(output_images, exist_ok=True)
os.makedirs(output_masks, exist_ok=True)
os.makedirs(dt_name_full + '/' + version, exist_ok=True)

# Path to the directory containing the images and labels
# segments_dir = 'segments/ale_phantom3/v0.2'
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
shutil.rmtree('ale')
if os.path.exists(f'export_coco-panoptic_{user}_{dt_name}_{version}.json'):
    os.remove(f'export_coco-panoptic_{user}_{dt_name}_{version}.json')

print("Processing complete. Images and masks moved to their respective folders.")
shutil.rmtree('ds/train', ignore_errors=True)
shutil.rmtree('ds/validation', ignore_errors=True)
shutil.rmtree('ds/test', ignore_errors=True)

time.sleep(2)

# Constants
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
SHUFFLE_IMAGES = True

OUTPUT_IMAGES_DIR = 'output/images'
OUTPUT_MASKS_DIR = 'output/masks'
MASKS_FILENAME_SUFFIX = 'label_ground-truth_coco-panoptic'

BONE_COLOR = (0, 113, 188)
MUSCLE_LAYER1_COLOR = (216, 82, 24)
MUSCLE_LAYER2_COLOR = (236, 176, 31)

# Ensure the output directories exist
assert os.path.exists(OUTPUT_IMAGES_DIR), "Output images directory does not exist."
assert os.path.exists(OUTPUT_MASKS_DIR), "Output masks directory does not exist."

# Remove any existing phantom_dataset directory
shutil.rmtree('phantom_dataset', ignore_errors=True)

# Create directories for train, validation, and test sets
for split in ['train', 'validation', 'test']:
    os.makedirs(f'ds/{split}/images', exist_ok=True)
    os.makedirs(f'ds/{split}/masks_muscle_layer1', exist_ok=True)
    os.makedirs(f'ds/{split}/masks_muscle_layer2', exist_ok=True)
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

def create_masks_for_colors(mask_path, muscle_layer1_save_path, muscle_layer2_save_path, bone_save_path):
    mask_image = Image.open(mask_path).convert('RGB')
    muscle_layer1_mask = Image.new('L', mask_image.size)
    muscle_layer2_mask = Image.new('L', mask_image.size)
    bone_mask = Image.new('L', mask_image.size)

    pixels = mask_image.load()
    muscle_layer1_pixels = muscle_layer1_mask.load()
    muscle_layer2_pixels = muscle_layer2_mask.load()
    bone_pixels = bone_mask.load()

    for y in range(mask_image.size[1]):
        for x in range(mask_image.size[0]):
            if pixels[x, y] == MUSCLE_LAYER1_COLOR:
                muscle_layer1_pixels[x, y] = 255
            elif pixels[x, y] == MUSCLE_LAYER2_COLOR:
                muscle_layer2_pixels[x, y] = 255
            elif pixels[x, y] == BONE_COLOR:
                bone_pixels[x, y] = 255

    muscle_layer1_mask.save(muscle_layer1_save_path)
    muscle_layer2_mask.save(muscle_layer2_save_path)
    bone_mask.save(bone_save_path)

def process_and_copy_files(file_list, split):
    for i, file in enumerate(file_list):
        base_filename = os.path.splitext(file)[0]
        image_src_path = os.path.join(OUTPUT_IMAGES_DIR, file)
        mask_src_path = os.path.join(OUTPUT_MASKS_DIR, f'{base_filename}_{MASKS_FILENAME_SUFFIX}.png')

        new_image_name = f'{i}.png'
        new_muscle_layer1_mask_name = f'{i}.png'
        new_muscle_layer2_mask_name = f'{i}.png'
        new_bone_mask_name = f'{i}.png'

        if os.path.exists(image_src_path) and os.path.exists(mask_src_path):
            shutil.copy(image_src_path, os.path.join(f'{split}/images', new_image_name))
            create_masks_for_colors(
                mask_src_path,
                os.path.join(f'{split}/masks_muscle_layer1', new_muscle_layer1_mask_name),
                os.path.join(f'{split}/masks_muscle_layer2', new_muscle_layer2_mask_name),
                os.path.join(f'{split}/masks_bone', new_bone_mask_name)
            )

        print(f'{split} organization: {round(100 * (i + 1) / len(file_list))} %')

# Process and copy the files to the respective directories
process_and_copy_files(train_files, 'ds/train')
process_and_copy_files(val_files, 'ds/validation')
process_and_copy_files(test_files, 'ds/test')

print("Dataset split into train, validation, and test sets with renamed files and separate masks for muscle layers and bone.")
shutil.rmtree('output', ignore_errors=True)
