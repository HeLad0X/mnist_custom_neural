import cupy as cp
from config import path_obj, ParamConfig
import struct

def read_mnist_images(datatype='train'):
    if datatype == 'train':
        image_path = path_obj.train_images_data()
        label_path = path_obj.train_images_label()
    else:
        image_path = path_obj.test_images_data()
        label_path = path_obj.test_images_label()
        
    with open(image_path, 'rb') as f:
        # Read the magic number and the number of images
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        # Read the image data
        images = cp.fromfile(f, dtype=cp.uint8).reshape(num_images, rows, cols)
        images = images.reshape(num_images, rows * cols)
    
    with open(label_path, 'rb') as f:
        # Read the magic number and the number of labels
        magic, num_labels = struct.unpack(">II", f.read(8))
        # Read the label data
        labels = cp.fromfile(f, dtype=cp.uint8)
        
    return images.T / ParamConfig.SCALE_FACTOR, labels


def get_custom_model_folder():
    return path_obj.custom_model_folder()

def get_custom_model_path():
    return path_obj.custom_model_path()

def is_model_generated():
    return path_obj.is_model_generated()


def init_params(size=28*28):
    W1 = cp.random.rand(ParamConfig.HIDDEN_NEURONS,size) - 0.5
    b1 = cp.random.rand(ParamConfig.HIDDEN_NEURONS,1) - 0.5
    W2 = cp.random.rand(ParamConfig.OUTPUT_NEURONS,ParamConfig.HIDDEN_NEURONS) - 0.5
    b2 = cp.random.rand(ParamConfig.OUTPUT_NEURONS,1) - 0.5
    return W1, b1, W2, b2


def ReLU(Z):
    return cp.maximum(0, Z)


def ReLU_derivative(Z):
    return (Z > 0).astype(int)


def softmax(Z):
    exp = cp.exp(Z - cp.max(Z))
    return exp / exp.sum(axis=0)


def one_hot(Y, num_classes=10):
    # Number of examples
    m = Y.shape[0]
    
    # Create a zero matrix of shape (num_classes, m)
    one_hot_matrix = cp.zeros((num_classes, m))
    
    # Use numpy's advanced indexing to set the appropriate elements to 1
    one_hot_matrix[Y, cp.arange(m)] = 1
    
    return one_hot_matrix


def compute_cost(A2, Y):
    m = Y.shape[0]  # Number of examples
    # Compute cross-entropy loss
    cost = -cp.sum(one_hot(Y) * cp.log(A2)) / m
    return cost
