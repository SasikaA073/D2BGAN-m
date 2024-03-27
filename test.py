import zipfile
import config


# unzip file 
def unzip_file(zip_file_path, extract_to):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        
# unzip_file("LOLdataset.zip", config.DATASET_DIR)

import os


def remove_non_image_files(directory):
    print("Processing directory:", directory)
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if not (file.endswith(".jpg") or file.endswith(".png")):
                print("Removing:", file_path)
                os.remove(file_path)
                
def print_non_image_files(directory):
    print("Processing directory:", directory)
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if not (file.endswith(".jpg") or file.endswith(".png")):
                print("Non-image file:", file_path)

# Example usage:
print_non_image_files(config.DATASET_DIR)

# Example usage:
remove_non_image_files(config.DATASET_DIR) 

# Example usage:
print("After")
print_non_image_files(config.DATASET_DIR)