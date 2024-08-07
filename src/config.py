import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2

#####################################################################################

# Device selection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#####################################################################################

# Data paths
DATA_DIR = '/kaggle/input/anime-sketch-colorization-pair/data'
TRAIN_DATASET_PATH = f'{DATA_DIR}/train'
TEST_DATASET_PATH = f'{DATA_DIR}/val'

#####################################################################################
  
# Note: It should be greater than 2048 to run FID evaluation
NUMBER_OF_IMAGES_TO_TEST = 2048 

EPOCH_MODEL_SAVE_PATH = "/kaggle/working/models"
INPUT_SAVE_DIRECTORY_PATH = "/kaggle/working/inputs"
TARGET_SAVE_DIRECTORY_PATH = "/kaggle/working/targets"
OUTPUT_SAVE_DIRECTORY_PATH = "/kaggle/working/outputs"

#####################################################################################

# Image properties
IMAGE_SIZE = 256

# Transformations for both input and target images
TRANSFORM_BOTH_IMAGES = A.Compose(
    [A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),],
    additional_targets={"image0": "image"},
)
# Transformations for input images only
TRANSFORM_INPUT_IMAGE_ONLY = A.Compose(
    [ 
        # Normalizes the pixel values of the image
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,), 
        # Convert image to PyTorch tensor
        ToTensorV2(),               
    ]
)
# Transformations for target images only
TRANSFORM_TARGET_IMAGE_ONLY = A.Compose(
    [
        # Normalizes the pixel values of the image
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        # Convert image to PyTorch tensor
        ToTensorV2(),
    ]
)

#####################################################################################

# Training parameters
LEARNING_RATE = 2e-4
TRAIN_BATCH_SIZE = 16
TRAIN_EPOCHS_COUNT = 50

# Hyperparameter controlling L1 loss in the generator loss function.
L1_LAMBDA = 100

#####################################################################################

#models and checkpoint will be saved in iteration of this.
ITERATION_TO_SAVE_MODEL = 5

LOAD_MODEL = True
SAVE_MODEL = True

# File names to save for discriminator and generator models
CHECKPOINT_DISC_FILE_NAME = "disc.pth.tar"
CHECKPOINT_GEN_FILE_NAME = "gen.pth.tar"

#####################################################################################

if __name__ == "__main__":
    print("Config module loaded.")