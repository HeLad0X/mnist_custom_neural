import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_kaggle_dataset():
    """Download the Kaggle dataset for hojjatk/mnist-dataset."""
    """kaggle datasets download -d jcprogjava/handwritten-digits-dataset-not-in-mnist"""
    dataset = 'jcprogjava/handwritten-digits-dataset-not-in-mnist'
    output_path = 'test_data/mnist_img'

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
    download_kaggle_dataset()
