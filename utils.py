import random, torch, os, numpy as np
import torch.nn as nn
import copy, zipfile 

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr, device="cuda"):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location= device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# unzip file 
def unzip_file(zip_file_path, extract_to):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        
def remove_non_image_files(directory):
    """
    Removes files that are not images from the directory 
    including in the subdirectories
    """
    print("Processing directory:", directory)
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if not (file.endswith(".jpg") or file.endswith(".png")):
                print("Removing:", file_path)
                os.remove(file_path)
                
def print_non_image_files(directory):
    """
    Print all files that are not images from the directory including in the subdirectories
    """
    print("Processing directory:", directory)
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if not (file.endswith(".jpg") or file.endswith(".png")):
                print("Non-image file:", file_path)