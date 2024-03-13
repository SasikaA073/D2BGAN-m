import os
import torch
from torchvision.transforms import Compose, RandomCrop, ToTensor, Normalize
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = os.cpu_count()

TRAIN_DIR = "LOLdataset/our485/"
VAL_DIR = "LOLdataset/eval15/"

LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10

LOAD_MODEL = False
SAVE_MODEL = True

CHECKPOINT_GEN_DARK = "gen_dark.pth.tar"
CHECKPOINT_GEN_BRIGHT = "gen_bright.pth.tar"

# Discriminators 
CHECKPOINT_CRITIC_DARK_C1 = "critic_dark_c1.pth.tar"
CHECKPOINT_CRITIC_DARK_C2 = "critic_dark_c2.pth.tar"
CHECKPOINT_CRITIC_DARK_T = "critic_dark_t.pth.tar"
CHECKPOINT_CRITIC_DARK_E = "critic_dark_e.pth.tar"

CHECKPOINT_CRITIC_BRIGHT_C1 = "critic_bright_c1.pth.tar"
CHECKPOINT_CRITIC_BRIGHT_C2 = "critic_bright_c2.pth.tar"
CHECKPOINT_CRITIC_BRIGHT_T = "critic_bright_t.pth.tar"
CHECKPOINT_CRITIC_BRIGHT_E = "critic_bright_e.pth.tar"

BATCH_SIZE = 1
LEARNING_RATE = 1e-5
NUM_EPOCHS = 10

TRAIN_IMG_SIZE = 256

DATASET_URL = 'https://docs.google.com/uc?export=download&id=1E1rGvhzu0oWMuake4NBDbrKWZOVzDrFP'
DATASET_ZIP_NAME = 'LOL_dataset.zip'

BLUR_SIGMA_1 = 2.0
BLUR_SIGMA_2 = 1.0

# Configuration setup from the D2BGAN paper

# How they did in the D2BGAN paper
LEARNING_RATE = 0.0002

train_transforms = Compose([
    RandomCrop(size=TRAIN_IMG_SIZE),
    ToTensor(),  # This will convert the PIL Image to a PyTorch tensor
    ToTensorV2(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
