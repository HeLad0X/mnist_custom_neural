import cupy as cp
from preprocess_non_mnist import get_images_labels_array_test
from src.evaluate import evaluate
from src.train import start_training


def split_train_test(images_array, labels, ratio = [0.7, 0.3]):
    size = images_array.shape[1]

    train_images = images_array.T[:int(ratio[0] * size)]
    train_labels = labels[:int(ratio[0] * size)]
    test_images = images_array.T[-int(ratio[1] * size):]
    test_labels = labels[-int(ratio[1] * size):]

    return train_images.T, train_labels, test_images.T, test_labels

def train_non_mnist(train_image, train_label):
    start_training(train_image, train_label)

def evaluate_non_mnist(test_image, test_label):
    evaluate(test_image, test_label)

def train_test_non_mnist():
    images_array, labels = get_images_labels_array_test(test_count=None)
    train_image, train_label, test_image, test_label = split_train_test(images_array, labels)

    train_non_mnist(train_image, train_label)
    evaluate_non_mnist(test_image, test_label)

if __name__ == '__main__':
    train_test_non_mnist()