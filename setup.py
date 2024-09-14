from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()

# Define setup configuration
setup(
    name="Neural Network from scratch",
    version="0.1",
    description="A module for setting up the environment",
    author="Shiraz Ahmad",
    author_email="shirazhambi786@gmail.com",
    packages=find_packages(include=['*']),
    install_requires=read_requirements(),
    
)