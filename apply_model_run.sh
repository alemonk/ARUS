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

python3 src/segm/run_model.py
python3 src/recon/us_reconstruction.py
python3 src/recon/plot_pointcloud.py

source deactivate