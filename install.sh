#!/bin/bash

python3 -m venv myenv
source myenv/bin/activate

pip3 install opencv-python
pip3 install matplotlib
pip3 install keras
pip3 install tensorflow
pip3 install torch torchvision
pip3 install -U scikit-learn
pip3 install segmentation_models_pytorch
