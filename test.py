import os
import shutil
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from config import (
    EPOCH_MODEL_SAVE_PATH, 
    INPUT_SAVE_DIRECTORY_PATH,
    ITERATION_TO_SAVE_MODEL, 
    NUMBER_OF_IMAGES_TO_TEST, 
    OUTPUT_SAVE_DIRECTORY_PATH, 
    TARGET_SAVE_DIRECTORY_PATH, 
    TEST_DATASET_PATH, 
    TRAIN_EPOCHS_COUNT
)
from dataset import AnimePairDataset
from gans.generator import Generator


#####################################################################################

def delete_and_make_dir(directory_path):
    # Delete directory_path if it exists
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    # Create directory
    os.makedirs(directory_path, exist_ok=True)

#####################################################################################

def create_necessary_directories(input_dir, target_dir, output_dir):
    '''Create directories for all directories (input_dir, target_dir, output_dir)'''
    delete_and_make_dir(input_dir)
    delete_and_make_dir(target_dir)
    delete_and_make_dir(output_dir)

#####################################################################################

def load_generator(model_path):
    '''Load model to generator'''
    gen_model = Generator(in_channels=3, features=64)
    gen_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Load the saved model weights
    gen_model.eval()
    return gen_model

#####################################################################################

def save_inputs_and_targets(input_dir, target_dir, output_dir, test_loader, total_images_to_save):
    '''Save inputs and target images in respective directories'''
    create_necessary_directories(input_dir, target_dir, output_dir)
    total = 0
    # Iterate through test loader
    for idx, (input_image, target_image) in enumerate(test_loader):
        total_images_in_batch = input_image.shape[0]
        # Denormalized input and target images
        input_image_denormalized = input_image * 0.5 + 0.5
        target_image_denormalized = target_image * 0.5 + 0.5
        # iterate through images in the batch
        for item_count_in_batch in range(total_images_in_batch):
            image_index = idx * total_images_in_batch + item_count_in_batch + 1 
            filename = f"{image_index:03d}.png"
            save_image(input_image_denormalized[item_count_in_batch], os.path.join(input_dir, filename))
            save_image(target_image_denormalized[item_count_in_batch], os.path.join(target_dir, filename))
        # ensure that batch is within the desired number of total images to save in test
        total += total_images_in_batch
        if total >= total_images_to_save:
            break  
    print(f"XXXXXXXXXXXXXXXXXXXXX {total}")

#####################################################################################

def estimate_ouput_and_save(output_dir, model_path, test_loader, total_images_to_save):
    '''Estimate output using model path and save it to output_dir'''
    delete_and_make_dir(output_dir)
    # load model and get generator
    gen_model = load_generator(model_path)
    # Iterate through test loader
    total = 0
    for idx, (input_image, _) in enumerate(test_loader):
        total_images_in_batch = input_image.shape[0]
        with torch.no_grad():
            # predict output
            generated_image = gen_model(input_image)
        # Denormalized generated images
        generated_image_denorm = generated_image * 0.5 + 0.5
        # iterate through images in the batch
        for item_count_in_batch in range(total_images_in_batch):
            image_index = idx * total_images_in_batch + item_count_in_batch + 1 
            filename = f"{image_index:03d}.png"  # Format to ensure three digits with leading zeros
            save_image(generated_image_denorm[item_count_in_batch], os.path.join(output_dir, filename))
        # ensure that batch is within the desired number of total images to save in test
        total += total_images_in_batch
        if total >= total_images_to_save:
            break  
    print(f"XXXXXXXXXXXXXXXXXXXXX {total}")

#######################################################################################

def test_and_save_outputs_using_last_model(batch_size = 16):
    print("#### Running Test using last saved model ####")
    # Test data loader
    test_loader = DataLoader(AnimePairDataset(TEST_DATASET_PATH), batch_size=batch_size, shuffle=False)
    # Save inputs and targets
    save_inputs_and_targets(
        input_dir = INPUT_SAVE_DIRECTORY_PATH, 
        target_dir = TARGET_SAVE_DIRECTORY_PATH, 
        output_dir = f"{OUTPUT_SAVE_DIRECTORY_PATH}/finals", 
        test_loader = test_loader, 
        total_images_to_save = NUMBER_OF_IMAGES_TO_TEST
    )
    # Predict output and save the input in output directories
    estimate_ouput_and_save(
        output_dir = f"{OUTPUT_SAVE_DIRECTORY_PATH}/finals", 
        model_path = f'{EPOCH_MODEL_SAVE_PATH}/epoch_{TRAIN_EPOCHS_COUNT}.pth', 
        test_loader = test_loader, 
        total_images_to_save = NUMBER_OF_IMAGES_TO_TEST
    )

#######################################################################################

def test_and_save_outputs_using_all_models(batch_size = 16):
    print("#### Running Test using all saved model ####")
    # Test data loader
    test_loader = DataLoader(AnimePairDataset(TEST_DATASET_PATH), batch_size=batch_size, shuffle=False)
    # Save inputs and targets
    save_inputs_and_targets(
        input_dir = INPUT_SAVE_DIRECTORY_PATH, 
        target_dir = TARGET_SAVE_DIRECTORY_PATH, 
        output_dir = f"{OUTPUT_SAVE_DIRECTORY_PATH}", 
        test_loader = test_loader, 
        total_images_to_save = NUMBER_OF_IMAGES_TO_TEST
    )
    # Predict output and save the input in output directories for each saved epoch
    for epoch in range(ITERATION_TO_SAVE_MODEL, TRAIN_EPOCHS_COUNT+1, ITERATION_TO_SAVE_MODEL):
        estimate_ouput_and_save(
            output_dir = f"{OUTPUT_SAVE_DIRECTORY_PATH}/{epoch}", 
            model_path = f'{EPOCH_MODEL_SAVE_PATH}/epoch_{epoch}.pth', 
            test_loader = test_loader, 
            total_images_to_save = NUMBER_OF_IMAGES_TO_TEST
        )

#####################################################################################

if __name__ == "__main__":
    print("Test module loaded.")