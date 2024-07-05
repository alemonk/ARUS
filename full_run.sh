#!/bin/bash

# Check if a choice was provided
if [ -z "$1" ]; then
  echo "Please select a configuration:"
  echo "1: Phantom3"
  echo "2: Forearm Ventral Final Clone"
  echo "3: Forearm Dorsal"
  read -p "Enter your choice: " choice
else
  choice=$1
fi
source set_config.sh $choice

source myenv/bin/activate

python3 ds/download_and_organize_dataset.py
python3 ds/data_augmentation.py
python3 segm/train_model.py
python3 segm/run_model.py
python3 recon/us_reconstruction.py
python3 recon/plot_pointcloud.py

source deactivate