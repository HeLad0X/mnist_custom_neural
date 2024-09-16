import os
import cupy as cp
import numpy as np
import cv2
from pathlib import Path
from download_non_mnist import download_kaggle_dataset

def get_test_data_folder():
    cwd = os.getcwd()
    image_folder = Path(cwd, 'test_data')
    return image_folder

def get_all_image_path():
    folder_path = get_test_data_folder()
    png_files = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.png'):
                png_files.append(os.path.join(root, file))

    return png_files

def shuffle_paths_and_create_labels(images_path, count):
    # Shuffle all the paths to get a random set of data for testing
    images_path = np.array(images_path)
    np.random.shuffle(images_path)

    # Take just the count of images passed for testing purpose
    images_path = images_path[:count]
    
    labels = [None] * images_path.size
    for i in range(len(images_path)):
        label = images_path[i].split('/')[-2]
        labels[i] = label

    return images_path, labels

def convert_png_to_array(image_path):
    # Open the image and convert to grayscale
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Resize the image to 28x28 pixels
    image = cv2.resize(image, (28, 28))

    # Check if the image has an alpha channel (4th channel in RGBA)
    if len(image.shape) == 3 and image.shape[2] == 4:
        # Extract the alpha channel
        image = image[:, :, 3]

    # Convert image to a cupy array
    image_array = cp.array(image)

    # Normalize the pixel values
    image_array = image_array / float(255)

    # Return the image array
    return image_array

def convert_all_images(images_path, test_count):
    images_list = [convert_png_to_array(image_path=image) for image in images_path]
    images_array = cp.array(images_list)
    images_array = images_array.reshape(test_count, 28*28)

    return images_array.T

def get_images_labels_array_test(test_count=10000):
    all_images_path = get_all_image_path()
    while len(all_images_path) == 0:
        download_kaggle_dataset()
    
    if test_count is None or test_count == 0:
        test_count = len(all_images_path)

    images_path, labels = shuffle_paths_and_create_labels(all_images_path, test_count)
    images_array = convert_all_images(images_path=images_path, test_count=test_count)
    labels = cp.array([int(num) for num in labels])

    return images_array, labels
    

