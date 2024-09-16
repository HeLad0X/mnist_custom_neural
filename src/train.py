# This script starts the hyperparameter tuning and saves the best parameters in a pickle dump file
import pickle
import cupy as cp
import os
from src.model import start_hyperparameter_tuning, forward_propagation
from src.utils import read_mnist_images, get_custom_model_folder, get_custom_model_path, is_model_generated

def save_best_params(best_params):
    # Check if folder exists otherwise create one
    if not os.path.exists(get_custom_model_folder()):
        os.makedirs(get_custom_model_folder())
    
    model_path = get_custom_model_path()

    # Save model as pickle dump
    with open(model_path, 'wb') as file:
        pickle.dump(best_params, file)

    print('Model saved...')


def train_model(image_train, label_train):

    # Start hyperparameter tuning
    best_params = start_hyperparameter_tuning(image_train, label_train)

    return best_params


def start_training(image_train, label_train):
    # Check if the model exists:
    if is_model_generated():
        print('NN Model already generated')
        return
    
    print('Starting the model training...')
    best_params = train_model(image_train, label_train)

    # Saving the best parameters to a pickle file
    save_best_params(best_params)

def get_predictions(X, W1, b1, W2, b2):
    # Perform forward propagation to get the predictions
    _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    
    # Convert probabilities to predicted classes
    predictions = cp.argmax(A2, axis=0)
    
    return predictions

def load_model():
    if not is_model_generated():
        print('No pretrained model detected...')
        start_training()
    else:
        print('Pretrained model detected')

    best_params = None
    with open(get_custom_model_path(), 'rb') as file:
        best_params = pickle.load(file)

    return best_params

if __name__ == '__main__':
    # Importing the required datasets
    image_train, label_train = read_mnist_images()
    start_training(image_train, label_train)
