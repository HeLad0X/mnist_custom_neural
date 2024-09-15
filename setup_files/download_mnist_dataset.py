import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def is_dataset_complete(output_path):
    # Check if the main dataset files are present and not empty
    expected_files = [
        't10k-images-idx3-ubyte/t10k-images-idx3-ubyte',
        't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte',
        'train-images-idx3-ubyte/train-images-idx3-ubyte',
        'train-labels-idx1-ubyte/train-labels-idx1-ubyte',

        't10k-images.idx3-ubyte',
        't10k-labels.idx1-ubyte',
        'train-images.idx3-ubyte',
        'train-labels.idx1-ubyte'
    ]
    
    for file in expected_files:
        file_path = os.path.join(output_path, file)
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            print(file_path)
            return False
    return True

def download_kaggle_dataset():
    """Download the Kaggle dataset for hojjatk/mnist-dataset."""
    dataset = 'hojjatk/mnist-dataset'
    output_path = 'data/mnist'

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Download the dataset using kaggle.api
    try:
        api.dataset_download_files(dataset, path=output_path)
        print(f"Downloaded {dataset} to {output_path}")
        
        # Unzipping the downloaded file
        zip_file_path = os.path.join(output_path, f'{dataset.split("/")[-1]}.zip')
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(output_path)
        os.remove(zip_file_path)  # Remove the zip file after extraction
        print(f"Unzipped the dataset to {output_path}")

    except Exception as e:
        print(f"An error occurred while downloading the Kaggle dataset: {e}")
        raise

# Run the download kaggle dataset function
if __name__ == '__main__':
    if not is_dataset_complete('data/mnist'):
        download_kaggle_dataset()
    else:
        print("Dataset is already complete.")
