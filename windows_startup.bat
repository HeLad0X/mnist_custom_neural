@echo off

REM Run the setup file
REM Step 1: Ensure 'wheel' is installed
pip install wheel

REM Step 2: Create the wheel distribution
python setup.py bdist_wheel

REM install whl file after the building is complete
pip install dist/your_package_name-version.whl


REM Run the download dataset file
echo Downloading dataset...
python download_mnist_dataset.py
