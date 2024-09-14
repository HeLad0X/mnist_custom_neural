#!/bin/bash

# Step 1: Ensure 'wheel' is installed
pip install wheel

# Step 2: Create the wheel distribution
python setup.py bdist_wheel

# install whl file after the building is complete
pip install dist/Neural_Network_from_scratch-0.1-py3-none-any.whl

# Run the download dataset file
echo "Downloading dataset..."
python3 download_mnist_dataset.py