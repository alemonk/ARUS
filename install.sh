#!/bin/bash

# Create a virtual environment
python3 -m venv myenv

# Activate the virtual environment
source myenv/bin/activate

# Check if the virtual environment is activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Virtual environment is activated."
else
    echo "Virtual environment is not activated. Exiting..."
    exit 1
fi

# Install all dependencies
pip3 install -r requirements.txt

# Error handling
if [ $? -eq 0 ]; then
    echo "All packages were installed successfully."
else
    echo "An error occurred during the installation. Exiting..."
    exit 1
fi
