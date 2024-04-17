# Anime Sketch Colorization Pix2Pix

### Abstract

Anime sketch colorization, a subset of image-to-image translation in computer vision, has witnessed notable
advancements, particularly through the application of Generative Adversarial Networks (GANs). This paper explores the efficacy of
Pix2Pix, a conditional GAN framework, in automating and enhancing the process of anime sketch colorization. We present a
comprehensive review of related works in this domain, highlighting the evolution from heuristic-based methods to deep learning
techniques like GANs. Specifically, we delve into the architecture and methodology of Pix2Pix, detailing its generator and discriminator
designs, loss functions, and experimental setups. Our experiments utilize a dataset of paired anime sketch and color images,
employing evaluation metrics such as Structural Similarity Index (SSIM), Fr ́echet Inception Distance (FID), and Peak Signal-to-Noise
Ratio (PSNR) to assess model performance. Results demonstrate progressive improvements in image quality and realism over training
epochs, with notable enhancements observed in FID and PSNR scores. Despite challenges such as training instability and fluctuating
SSIM scores, our findings underscore the potential of Pix2Pix in automating anime sketch colorization tasks, paving the way for further
research and advancements in this domain.

### Directory and Files

It consists of several Python files:

- `utils.py`: Provides utility functions used across different parts of the project.
- `dataset.py`: Defines the dataset class for loading and preprocessing the anime sketch images.
- `gans/discriminator.py`: Implements the discriminator network architecture.
- `gans/generator.py`: Implements the generator network architecture.
- `train.py`: Contains the code for training the Pix2Pix model.
- `test.py`: Contains the code for testing the trained model and saving predicted images.
- `eval.py`: Contains evaluation functions to assess the performance of the model.
- `main.py`: Main script to orchestrate the training, testing, and evaluation processes.
- `config.py`:
  - Contains configuration constants for the model training and evaluation.
  - _Dataset paths_: Multiple constants to save the dataset paths. Currently it contains the kaggle directory path because we trained our model in kaggle platform.
  - _Training configs_: includes constants such as number of training epochs, learning rate, L1 lambda, batch size.
  - _Augmentation configs_: includes transformation configs for both input and target images.
  - _Test configs_: includes total number of images to test, and paths to store input, target and output/predicted images while testing the model. These paths points to the kaggle directory as we tested our modle in the kaggle platform.
  - _Saving Model configs_: includes boolean constants which controls the decision of whether to save and load the model or not, name of checkpoints files to be saved for both generator and discriminator, and the iteration number in which the model will be saved.

### Installation

To run this project, ensure that Python is installed on your system along with the following dependencies:

- PyTorch
- torchvision
- NumPy
- Matplotlib
- scikit-image
- clean-fid
- albumentations
- albumentations.pytorch
- PIL
- tqdm

### Usage

If you want to execute training, tests and evaluation at once, you can simply import the main module and then call start function.

```python
import main as pix2pix
pix2pix.start()
```

Training actually saves the generator model for each iteration at the interval of `ITERATION_TO_SAVE_MODEL` which is set in config file. So, by default, test and the evaluation executes on all model saved models. So, for each saved model or the epoch interval, it saves the output folder, use them to evaluate the SSIM, PSNR and FID scores. Finally, the plot showing the scores over the epochs is saved.

If you want to use the last model only, then you can simply call the method `set_test_epoch_frequency` and pass `False`. It ensures that only the last model is used to predict the output and evaluate the model using SSIM, PSNR and FID evaluation metrics. No plot will be saved in this setting. It is important to note that training should be successful until the last epoch set in config because the test uses this value to get the last saved model.

```python
import main as pix2pix
pix2pix.set_test_epoch_frequency(False)
pix2pix.start()
```

If someone wishes to run in isolation, then

1. To train the model, run

   ```python
   import train
   train.train_pix2pix()
   ```

2. To test and save outputs for all model, run

   ```python
   import test
   test.test_and_save_outputs_using_all_models()
   ```

   Or, to test and save output of last model, run

   ```python
   import test
   test.test_and_save_outputs_using_last_model()
   ```

3. To evaluate all models, run

   ```python
   import eval
   eval.execute_evaluation_for_all_models()
   ```

   Or, to evaluate the last model, run

   ```python
   import eval
   eval.execute_evaluation_for_last_model()
   ```

Note that they use the utils.py and config.py internally. So, if you want to configure any settings, you should update the values according to your requirement in those scripts.

### Artifacts and Results

Even though the current code has a default value of 50 for training epoch count and 5 for iterative count interval to save the models, resulting in a total of 10 models from 5, 10, 15, 20, ... 50; we experimented for 150 epochs with an interval of 10 resulting in 15 models.

All related works and results our experiment are stored in the following Google Drive folder: [Computer Vision W2024 - Group 08](https://drive.google.com/drive/folders/17he-V1wXIaBfbVBEsyjr5rGmvnPmdTiX?usp=sharing).

Google Drive Link: [https://drive.google.com/drive/folders/17he-V1wXIaBfbVBEsyjr5rGmvnPmdTiX?usp=sharing](https://drive.google.com/drive/folders/17he-V1wXIaBfbVBEsyjr5rGmvnPmdTiX?usp=sharing).

The experiment results are:

1. The dataset we used is extremely large to keep in the drive. Therefore, we are directly giving the link to the dataset [Anime Sketch Colorization Pair](https://www.kaggle.com/datasets/ktaebum/anime-sketch-colorization-pair).
2. The checkpoints are stored in the directory `checkpoints`.
   - It contains `disc.pth.tar` and `gen.pth.tar` of discriminator and generator respectively, representing the model and optimizer state after epoch 150.
3. The models for each iteration are named `epoch_{i}_model.pth` where i ∈ {10, 20, 30, ... 150}.
   - For each model, SSIM, FID, and PSNR were calculated, and the trends were plotted and displayed in the report.
4. Tests were conducted for each model. We tested 2050 images from the validation dataset. It includes following three directories:
   1. `inputs`: contains 2050 sketch images used as input.
   2. `targets`: contains 2050 colored images respective to the sketch, used as target.
   3. `outputs`: It contains 15 directories (10, 20, 30, ... 150), each representing the saved epoch. Each directory contains 2050 predicted images using the model of that corresponding epoch.
