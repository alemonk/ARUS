import os
from PIL import Image, ImageOps
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from params import n_class, dt_name

# Define the directories
BASE_DIR = f'out/{dt_name}'
SPLITS = ['train', 'validation', 'test']

# Generate subdirectory names dynamically based on the number of classes
def generate_subdirs(num_classes):
    subdirs = ['images']
    for i in range(num_classes):
        subdirs.append(f'masks_class_{i}')
    return subdirs

SUBDIRS = generate_subdirs(n_class)

# Function to horizontally flip images in a directory
def flip_images_horizontally(directory):
    for subdir in SUBDIRS:
        subdir_path = os.path.join(directory, subdir)
        if not os.path.exists(subdir_path):
            continue

        for filename in os.listdir(subdir_path):
            if filename.endswith(('.jpg', '.png')):
                file_path = os.path.join(subdir_path, filename)
                image = Image.open(file_path)
                flipped_image = ImageOps.mirror(image)
                
                new_filename = f"{os.path.splitext(filename)[0]}_flipped{os.path.splitext(filename)[1]}"
                flipped_image.save(os.path.join(subdir_path, new_filename))
                print(f"Saved flipped image {new_filename} in {subdir_path}")

# Process each split
for split in SPLITS:
    split_path = os.path.join(BASE_DIR, split)
    flip_images_horizontally(split_path)

print("Flipping complete for all images in train, validation, and test sets.")
