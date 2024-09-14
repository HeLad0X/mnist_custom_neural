import os
import kaggle

def is_dataset_complete(output_path):
    # Check if the main dataset files are present and not empty
    expected_files = [
        'train.csv',   # Example file names, replace with actual expected files
        'test.csv'
    ]
    
    for file in expected_files:
        file_path = os.path.join(output_path, file)
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return False
    return True

def download_kaggle_dataset():
    dataset = 'hojjatk/mnist-dataset'
    output_path = 'data/mnist'
    
    # Check if the dataset is already downloaded and complete
    if is_dataset_complete(output_path):
        print("Dataset already downloaded and complete.")
        return
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Download the dataset
    try:
        kaggle.api.dataset_download_files(dataset, path=output_path, unzip=True)
        print(f"Downloaded {dataset} to {output_path}")
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        raise


# Run the download kaggle datset function
if __name__ == '__main__':
    download_kaggle_dataset()