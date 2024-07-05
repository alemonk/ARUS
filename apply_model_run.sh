#!/bin/bash

echo "Select configuration:"
echo "1: Phantom3"
echo "2: Forearm Ventral Final Clone"
echo "3: Forearm Dorsal"
read -p "Enter your choice: " choice

case $choice in
    1)
        export USER="alemonk"
        export DT_NAME="phantom3"
        export VERSION="v0.1"
        export LABEL_SET="ground-truth"
        export FILTER="labeled reviewed"
        export N_CLASS=1
        ;;
    2)
        export USER="ale"
        export DT_NAME="forearm-ventral-final-clone"
        export VERSION="v0.3"
        export LABEL_SET="ground-truth"
        export FILTER="labeled reviewed"
        export N_CLASS=3
        ;;
    3)
        export USER="ale"
        export DT_NAME="forearm-dorsal"
        export VERSION="v0.1"
        export LABEL_SET="ground-truth"
        export FILTER="labeled reviewed"
        export N_CLASS=1
        ;;
    *)
        echo "Invalid choice, exiting."
        exit 1
        ;;
esac

export INPUT_SEGMENTATION="ds/test-${DT_NAME}"
export OUTPUT_SEGMENTATION="segm/results-test-${DT_NAME}"
export OUTPUT_MODEL_TRAIN="segm/results-training-${DT_NAME}"
export MODEL_DIRECTORY="segm/model-${N_CLASS}class-${DT_NAME}.model"
export POSES_FILENAME="recon/img_pose.txt"
export IMG_FOLDER_PATH="${OUTPUT_SEGMENTATION}/output_segmentation"
export OUTPUT_POINTCLOUD_DIR="recon/pointclouds/${DT_NAME}"
export IMGS_HEIGHT_CM=6.0
export POINTCLOUD_FILENAMES=$(for ((i=0; i<$N_CLASS; i++)); do echo "recon/pointclouds/${DT_NAME}/${i}.txt"; done | tr '\n' ' ')

echo "Configuration set."

###########################################################################

source myenv/bin/activate

python3 segm/run_model.py
python3 recon/us_reconstruction.py
python3 recon/plot_pointcloud.py

source deactivate