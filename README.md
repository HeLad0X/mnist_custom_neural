
# Neural Network from Scratch

This project implements a neural network from scratch using basic Python libraries such as NumPy, CuPy, and Pandas, without relying on deep learning frameworks like PyTorch or TensorFlow. The neural network is trained on the MNIST dataset and consists of three layers: an input layer, a hidden layer, and an output layer.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The aim of this project is to gain a fundamental understanding of neural networks by building one from the ground up using only basic Python libraries. This approach allows for a deeper comprehension of core concepts such as forward propagation, backpropagation, and gradient descent.

This implementation uses the MNIST dataset, a widely recognized dataset of handwritten digits, to train the neural network. The network architecture consists of three layers:
- **Input Layer:** Takes in the pixel values of the MNIST images.
- **Hidden Layer:** Contains neurons that process the input features.
- **Output Layer:** Outputs the probabilities for each of the 10-digit classes (0-9).

## Features

- Custom neural network built using cupy.
- Implementation of forward and backward propagation.
- Gradient descent optimization and hyperparameter tuning.
- Support for various activation functions (ReLU, Sigmoid, ReLU_derivative).
- Evaluation metrics for performance assessment (accuracy, loss).

## Installation

1. **Clone the repository:**
    `git clone https://github.com/HeLad0X/mnist_custom_neural`

2. **Navigate to the project directory:**\
    `cd mnist_custom_neural`

3. **Install the CUDA Drivers:**
    The Script uses `cupy` instead of `numpy` to increase the program's computation speed using GPU.
    For Windows users, using WSL2: `https://docs.nvidia.com/cuda/wsl-user-guide/index.html`
    For Linux users: `https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#meta-packages`

    **Note:** You can still use `numpy` instead of `cupy`, change all occurrences of cupy with numpy.

4. **Install the required dependencies and the dataset(Creates a virtual environment directory '.env' by default):**
    On Windows use `startup_script.sh` using git bash
    On Linux use `chmod +x startup_script.sh` then run `startup_script.sh`
    
    **Note:** The script uses public API from Kaggle that needs to be set up by the user.
    See: https://www.kaggle.com/docs/api

## Usage

1. **Prepare the MNIST dataset:**
   The MNIST dataset will be automatically downloaded and preprocessed using the startup script and setup functions in the project.

2. **Configure the Neural Network:**
   Modify the network parameters such as the number of neurons in the hidden layer, learning rate, and the number of epochs in the `config.py` file.

3. **Train the Model:**
   Run the training script to start training the neural network on the MNIST dataset:
    \`\`\`bash
    python train.py
    \`\`\`

4. **Evaluate the Model:**
   After training, evaluate the model's performance using the evaluation script:
    \`\`\`bash
    python evaluate.py
    \`\`\`

## Project Structure

- `src/train.py`: The main script for training the neural network on the MNIST dataset.
- `src/evaluate.py`: Script to evaluate the trained neural network.
- `src/model.py`: Contains the implementation of the neural network class.
- `src/config.py`: Configuration file for setting hyperparameters such as learning rate and number of neurons in the hidden layer.
- `src/utils.py`: Utility functions for data preprocessing and other helper functions.
- `requirements.txt`: List of required Python packages.
- `data/`: Directory to store the MNIST dataset.
- `src/train_tensor`: File with a script that implements tensorflow and checks the accuracy of the dataset for cross-validation.
- `startup_script.sh`: File with a script that creates an env in the current directory and installs all the required Python libraries and the MNIST dataset.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements. Ensure that your code follows the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
