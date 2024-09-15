import os
from setuptools import setup, find_packages

def read_requirements():
    # Read the requirements.txt file and return a list of dependencies
    with open('requirements.txt') as f:
        return f.read().splitlines()

# Change the current directory to the parent directory (root) to include all files and folders
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")

# Read the contents of your README file for the long description
with open('README.md') as f:
    long_description = f.read()

setup(
    name="Custom-Neural-Network",
    version="0.1",
    description="A custom created Neural Network for mnist dataset from kaggle",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Shiraz Ahmad",
    author_email="shirazhambi786@gmail.com",
    packages=find_packages(),  # Automatically finds and includes all packages
    install_requires=read_requirements(),  # Include dependencies from requirements.txt
)
