# This script starts the hyperparameter tuning and saves the best parameters in a pickle dump file
import pickle
import cupy as cp
import os
from model import start_hyperparameter_tuning, forward_propagation
from utils import read_mnist_images, get_custom_model_folder, get_custom_model_path, is_model_generated

def save_best_params(best_params):
    # Check if folder exists otherwise create one
    if not os.path.exists(get_custom_model_folder()):
        os.makedirs(get_custom_model_folder())
    
    model_path = get_custom_model_path()

    # Save model as pickle dump
    with open(model_path, 'wb') as file:
        pickle.dump(best_params, file)

    print('Model saved...')


def train_model():
    # Importing the required datasets
    image_train, label_train = read_mnist_images()

    # Start hyperparameter tuning
    best_params = start_hyperparameter_tuning(image_train, label_train)

    return best_params


def start_training():
    # Check if the model exists:
    if is_model_generated():
        print('NN Model already generated')
        return
    
    print('Starting the model training...')
    best_params = train_model()

    # Saving the best parameters to a pickle file
    save_best_params(best_params)

def get_predictions(X, W1, b1, W2, b2):
    # Perform forward propagation to get the predictions
    _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    
    # Convert probabilities to predicted classes
    predictions = cp.argmax(A2, axis=0)
    
    return predictions

if __name__ == '__main__':
    start_training()