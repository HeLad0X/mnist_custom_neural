#!/bin/bash

# Determine the platform (Linux or Windows)
PLATFORM="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="linux"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    PLATFORM="windows"
fi

# Name of the virtual environment directory
VENV_DIR=".env"

# Function to create and activate the virtual environment
create_and_activate_venv() {
    # Check if the virtual environment directory exists
    if [ ! -d "$VENV_DIR" ]; then
        echo "Virtual environment '$VENV_DIR' not found. Creating one..."
        # Suppress the "scripts not in PATH" warning on Windows
        python3 -m venv $VENV_DIR
        if [ $? -ne 0 ]; then
            echo "Failed to create virtual environment. Please ensure Python and venv are installed."
            exit 1
        fi
        echo "Virtual environment '$VENV_DIR' created successfully."
    else
        echo "Virtual environment '$VENV_DIR' already exists."
    fi

    # Activate the virtual environment
    if [ "$PLATFORM" == "windows" ]; then
        # Use call to make sure this runs in the same command session
        call $VENV_DIR\\Scripts\\activate
    else
        source $VENV_DIR/bin/activate
    fi

    echo "Virtual environment '$VENV_DIR' activated."
}

# Step 1: Create and activate the virtual environment
create_and_activate_venv

# Set the Python and Pip commands for the virtual environment
if [ "$PLATFORM" == "windows" ]; then
    PYTHON_CMD="$VENV_DIR/Scripts/python"
    PIP_CMD="$VENV_DIR/Scripts/pip"
else
    PYTHON_CMD="$VENV_DIR/bin/python3"
    PIP_CMD="$VENV_DIR/bin/pip"
fi

# Step 2: Ensure 'wheel' is installed
$PIP_CMD install wheel==0.44.0

# Step 3: Create the wheel distribution
$PYTHON_CMD setup_files/setup.py bdist_wheel

# Step 4: Install the generated .whl file
$PIP_CMD install dist/Custom_Neural_Network-0.1-py3-none-any.whl

# Step 5: Run the download dataset file
echo "Downloading dataset..."
$PYTHON_CMD setup_files/download_mnist_dataset.py
