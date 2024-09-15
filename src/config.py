import os
from pathlib import Path

class PathConfig:
    def __init__(self) -> None:
        self.__data_path = Path(os.path.join(os.getcwd(), 'data/mnist'))
        self.__model_folder = Path(os.path.join(os.getcwd(), 'models'))
        self.__model_name='custom_nn_params.pkl'
        self.__model_generated = False

    def train_images_data(self):
        return Path(os.path.join(self.__data_path, 'train-images.idx3-ubyte'))
    
    def test_images_data(self):
        return Path(os.path.join(self.__data_path, 't10k-images.idx3-ubyte'))
    
    def train_images_label(self):
        return Path(os.path.join(self.__data_path, 'train-labels.idx1-ubyte'))
    
    def test_images_label(self):
        return Path(os.path.join(self.__data_path, 't10k-labels.idx1-ubyte'))
    
    def custom_model_path(self):
        return Path(os.path.join(self.__model_folder, self.__model_name))
    
    def custom_model_folder(self):
        return self.__model_folder
    
    def is_model_generated(self):
        if self.__model_generated:
            return True
        else:
            self.__model_generated = os.path.exists(self.custom_model_path())
            return self.__model_generated


class ParamConfig:
    # various parameters
    LEARNING_RATES = [0.1, 0.05, 0.01, 0.005, 0.001]
    EPOCHS = [10, 20, 30, 40, 50]
    HIDDEN_NEURONS = 64
    OUTPUT_NEURONS = 10
    BATCH_SIZE = 64
    SCALE_FACTOR = 255

path_obj = PathConfig()