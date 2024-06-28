# pip install segments-ai
from segments import SegmentsClient, SegmentsDataset
from segments.utils import export_dataset
import os
import shutil

dt_name = 'alemonk/forearm-ventral-short'
version = '0.1'

# Initialize a SegmentsDataset from the release file
client = SegmentsClient('f9a1e7f6dab2ca00db793d1e45ca06f40d10173e')
release = client.get_release(dt_name, 'v0.1')
dataset = SegmentsDataset(release, labelset='ground-truth', filter_by=['labeled', 'reviewed'])

# Export to COCO panoptic segmentation format
export_dataset(dataset, export_format='coco-panoptic')

output_images = 'output/images'
output_masks = 'output/masks'

# Create directories for images and masks
os.makedirs(output_images, exist_ok=True)
os.makedirs(output_masks, exist_ok=True)
os.makedirs(dt_name + '/' + version, exist_ok=True)

# Path to the directory containing the images and labels
# segments_dir = 'segments/alemonk_phantom3/v0.2'
segments_dir = 'segments/alemonk_forearm-ventral-short/v0.1'
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
shutil.rmtree('alemonk')
if os.path.exists('export_coco-panoptic_alemonk_forearm-ventral-short_v0.1.json'):
    os.remove('export_coco-panoptic_alemonk_forearm-ventral-short_v0.1.json')

print("Processing complete. Images and masks moved to their respective folders.")
