import os
from PIL import Image, ImageOps

# Define the directories
BASE_DIR = 'ds'
SPLITS = ['train', 'validation', 'test']
SUBDIRS = ['images', 'masks_muscle_layer1', 'masks_muscle_layer2', 'masks_bone']

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

print("Flipping complete for all images in test, train, and validation sets.")
