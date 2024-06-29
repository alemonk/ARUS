#!/bin/bash

# Define the python interpreter
PYTHON_INTERPRETER="./myenv/bin/python"

# Uncomment the following lines if you need to train the model first

# # DOWNLOAD DATASET FROM SEGMENTS.AI AND ORGANIZE IT INTO TRAIN, VALIDATION AND TEST
# $PYTHON_INTERPRETER "./ds/download_dataset.py"
# $PYTHON_INTERPRETER "./ds/organize_dataset.py"

# # EXTRACT FRAMES FROM FOREARM TEST DATASET
# $PYTHON_INTERPRETER "./ds/extract_frames.py"

# # TRAIN MODEL
# $PYTHON_INTERPRETER "./segm/train_model.py"

# RUN MODEL
$PYTHON_INTERPRETER "./segm/run_model.py"

# 3D RECONSTRUCTION
$PYTHON_INTERPRETER "./recon/us_reconstruction.py"
$PYTHON_INTERPRETER "./recon/plot_pointcloud.py"
