import cv2
import os
import re

def load_images(image_filenames, image_folder):
    images = []
    for filename in image_filenames:
        print(filename)
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    return images

def stitch_images(images):
    imageStitcher = cv2.Stitcher_create()
    error, stitched_image = imageStitcher.stitch(images)
    if not error:
        cv2.imwrite('stitching/stitched_output.png', stitched_image)
        return error, stitched_image
    else:
        print('Error! No matches found?')
        return error, []

def numerical_sort(value):
    # Extract numeric part and original filename as a tuple
    parts = re.findall(r'\d+', value)
    if parts:
        return int(parts[0]), value
    return 0, value

# Load images
image_folder = 'stitching/test_images/test_pics'
image_filenames = sorted(os.listdir(image_folder), key=numerical_sort)

images = load_images(image_filenames, image_folder)
error, stitched_image = stitch_images(images)
