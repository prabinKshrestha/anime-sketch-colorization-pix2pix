import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from cleanfid import fid


from config import DEVICE, INPUT_SAVE_DIRECTORY_PATH, ITERATION_TO_SAVE_MODEL, NUMBER_OF_IMAGES_TO_TEST, OUTPUT_SAVE_DIRECTORY_PATH, TARGET_SAVE_DIRECTORY_PATH, TRAIN_EPOCHS_COUNT

#####################################################################################

# Function to load images
def load_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    return np.array(image)

#####################################################################################

# Function to calculate SSIM from grayscale images
def calculate_SSIM(y, yhat):
    ssim_score = ssim(y, yhat, data_range=y.max() - y.min())
    return ssim_score

#####################################################################################

# Function to calculate PSNR from grayscale images
def calculate_PSNR(y, yhat):
    psnr_score = psnr(y, yhat, data_range=y.max() - y.min())
    return psnr_score

#####################################################################################

# Function to display images
def display_images(image_paths):
    fig, axes = plt.subplots(3, 15, figsize=(15, 3))
    for i, paths in enumerate(image_paths):
        for j, path_ind in enumerate(paths):
            ax_inner = axes[j, i]
            image = Image.open(path_ind)
            ax_inner.imshow(image)
            ax_inner.axis('off')
    plt.show()

#####################################################################################

def evaluate(input_dir, target_dir, output_dir, total_image, show_images=True):
    # Prepare paths for input, output, and target images
    input_image_paths = [os.path.join(input_dir, f"{i:03d}.png") for i in range(1, total_image)]
    target_image_paths = [os.path.join(target_dir, f"{i:03d}.png") for i in range(1, total_image)]
    output_image_paths = [os.path.join(output_dir, f"{i:03d}.png") for i in range(1, total_image)]
    # if show image is true, display or save sample images
    if show_images:
        num_images = 15
        # Assuming input_image_paths, output_image_paths, and target_image_paths are lists of image paths with the same length
        all_image_paths = list(zip(input_image_paths, output_image_paths, target_image_paths))
        random.shuffle(all_image_paths)
        random_image_paths = all_image_paths[:num_images]
        display_images(random_image_paths)

     # Load and calculate SSIM for each image pair
    total_ssim = 0.0
    total_psnr= 0.0
    for target_path, output_path in zip(target_image_paths, output_image_paths):
        target_image = load_image(target_path)
        output_image = load_image(output_path)
        total_ssim += calculate_SSIM(target_image, output_image)
        total_psnr += calculate_PSNR(target_image, output_image)

    # Calculate average SSIM for the epoch
    average_ssim = total_ssim / len(input_image_paths)
    print(f"Mean SSIM Score: {average_ssim}")
    average_psnr = total_psnr / len(input_image_paths)
    print(f"Mean PSNR Score: {average_psnr}")
    fid_score = fid.compute_fid(output_dir, target_dir, device=DEVICE, num_workers=0)
    print(f"FID Score: {fid_score}")

    return average_ssim, average_psnr, fid_score
    

#####################################################################################

def execute_evaluation_for_last_model():
    print("#### Running Evaluation for last model ####")
    evaluate(
        input_dir = INPUT_SAVE_DIRECTORY_PATH, 
        target_dir = TARGET_SAVE_DIRECTORY_PATH, 
        output_dir = f"{OUTPUT_SAVE_DIRECTORY_PATH}/finals", 
        total_image = NUMBER_OF_IMAGES_TO_TEST,
    )

###########################################################################

def plot_evaluation_metrics(ssim_scores, psnr_scores, fid_scores):
    print("#### Plotting Evaluation Metrics ####")
    # Define epochs
    number_of_epochs = range(ITERATION_TO_SAVE_MODEL, TRAIN_EPOCHS_COUNT+1, ITERATION_TO_SAVE_MODEL)

    # Create a figure and an array of subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot SSIM scores
    axes[0].plot(number_of_epochs, ssim_scores, label='SSIM')
    axes[0].set_xlabel('Number of Epochs')
    axes[0].set_ylabel('Mean SSIM Score')
    axes[0].set_title('Mean SSIM Scores over Epochs')

    # Plot PSNR scores
    axes[1].plot(number_of_epochs, psnr_scores, label='PSNR')
    axes[1].set_xlabel('Number of Epochs')
    axes[1].set_ylabel('Mean PSNR Score')
    axes[1].set_title('Mean PSNR Scores over Epochs')

    # Plot FIDs scores
    axes[2].plot(number_of_epochs, fid_scores, label='FIDs')
    axes[2].set_xlabel('Number of Epochs')
    axes[2].set_ylabel('FID Score')
    axes[2].set_title('FID Scores over Epochs')

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig('evaluation_plot.png')

def execute_evaluation_for_all_models():
    print("#### Running Evaluation for all models ####")
    ssim_collection, psnr_collection, fid_score_collection = [], [], []
    for epoch in range(ITERATION_TO_SAVE_MODEL, TRAIN_EPOCHS_COUNT+1, ITERATION_TO_SAVE_MODEL):
        print(f"Evaluating for epoch {epoch}")
        average_ssim, average_psnr, fid_score = evaluate(
            input_dir = INPUT_SAVE_DIRECTORY_PATH, 
            target_dir = TARGET_SAVE_DIRECTORY_PATH, 
            output_dir = f"{OUTPUT_SAVE_DIRECTORY_PATH}/{epoch}", 
            total_image = NUMBER_OF_IMAGES_TO_TEST,
        )
        ssim_collection.append(average_ssim)
        psnr_collection.append(average_psnr)
        fid_score_collection.append(fid_score)
    
    # plot evaluation metrics
    plot_evaluation_metrics(ssim_collection, psnr_collection, fid_score_collection)

#####################################################################################

if __name__ == "__main__":
    print("Eval module loaded.")