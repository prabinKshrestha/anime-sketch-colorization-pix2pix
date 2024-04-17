import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import TRANSFORM_BOTH_IMAGES, TRANSFORM_INPUT_IMAGE_ONLY, TRANSFORM_TARGET_IMAGE_ONLY, TRAIN_DATASET_PATH

#####################################################################################


class AnimePairDataset(Dataset):
    '''Anime Pair Dataset'''

    def __init__(self, dataset_dir, both_transform=TRANSFORM_BOTH_IMAGES, input_transform=TRANSFORM_INPUT_IMAGE_ONLY, output_transform=TRANSFORM_TARGET_IMAGE_ONLY):
        '''Initialize the dataset with necessary transformations'''
        self.dataset_dir = dataset_dir  # Directory of the dataset
        self.list_images = os.listdir(self.dataset_dir)  # List of images in the dataset directory
        self.both_transform = both_transform  # Transformations which will be applied to both input and target images
        self.input_transform = input_transform  # Transformations which will be applied only to input images
        self.output_transform = output_transform  # Transformations which will be applied only to target images

    def __len__(self):
        '''Return the total number of images in the dataset'''
        return len(self.list_images)

    def __getitem__(self, index):
        '''Get a pair of input and target images by index'''
        orig_image = self._get_image_from_directory(index)  # Load original image from the dataset directory
        input_image, target_image = self._split_image_into_input_and_target(orig_image)  # Split the original image into input and target images
        input_image, target_image = self._apply_transformations(input_image, target_image)  # Apply transformations to input and target images
        return input_image, target_image
    
    def _get_image_from_directory(self, index):
        '''Load image from the dataset folder at the index'''
        img_file = self.list_images[index]
        img_path = os.path.join(self.dataset_dir, img_file)
        return np.array(Image.open(img_path))
    
    def _split_image_into_input_and_target(self, full_image):
        '''Split full_image into input and target image'''
        w = full_image.shape[1] // 2
        target_image = full_image[:, :w, :]
        input_image = full_image[:, w:, :]
        return input_image, target_image
    
    def _apply_transformations(self, input_image, target_image):
        '''Apply transformation to input and target image'''
        augmentations = self.both_transform(image=input_image, image0=target_image)
        input_image, target_image = augmentations["image"], augmentations["image0"]
        input_image = self.input_transform(image=input_image)["image"]
        target_image = self.output_transform(image=target_image)["image"]
        return input_image, target_image

#####################################################################################

def show_example_using_dataloader_test():    
    '''Test AnimePairDataset is working by using it to show image'''
    total_example = 5
    x, y = next(iter(DataLoader(AnimePairDataset(TRAIN_DATASET_PATH), batch_size=total_example, shuffle=True)))
    fig, axes = plt.subplots(nrows=2, ncols=total_example, figsize=(total_example*2, 4))
    for j in range(total_example):
        axes[0, j].imshow(x[j].permute(1, 2, 0).cpu())
        axes[0, j].axis('off')
        axes[1, j].imshow(y[j].permute(1, 2, 0).cpu())
        axes[1, j].axis('off')
    plt.savefig(f'test_anime_pair_dataset.png')

#####################################################################################

if __name__ == "__main__":
    show_example_using_dataloader_test()

#####################################################################################