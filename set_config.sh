#!/bin/bash

# Accept a configuration choice as an argument
choice=$1

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

export INPUT_SEGMENTATION="out/${DT_NAME}/test-unseen_data"
export OUTPUT_SEGMENTATION="out/${DT_NAME}/results-test"
export OUTPUT_MODEL_TRAIN="out/${DT_NAME}/results-training"
export MODEL_DIRECTORY="out/${DT_NAME}/model-${N_CLASS}class.model"
export POSES_FILENAME="src/recon/img_pose.txt"
export IMG_FOLDER_PATH="${OUTPUT_SEGMENTATION}/output_segmentation"
export OUTPUT_POINTCLOUD_DIR="out/${DT_NAME}/pointclouds"
export IMGS_HEIGHT_CM=6.0
export POINTCLOUD_FILENAMES=$(for ((i=0; i<$N_CLASS; i++)); do echo "${OUTPUT_POINTCLOUD_DIR}/${i}.txt"; done | tr '\n' ' ')

echo "Configuration set for choice $choice."
