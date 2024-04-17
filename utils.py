import os
import matplotlib.pyplot as plt
import torch

from config import DEVICE

####################################################################################

def show_examples(gen, val_loader, epoch):
    ''' Show example with generator gen, of dataloader val_loader for epoch'''
    print("*** Showing Examples ***")
    x, y = next(iter(val_loader))
    x, y = x.to(DEVICE), y.to(DEVICE)
    total_example = len(y)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization
        # Plot original images (x), ground truth images (y), and generated fake images (y_fake)
        fig, axes = plt.subplots(nrows=3, ncols=total_example, figsize=(total_example*2, 3))
        for j in range(total_example):
            axes[0, j].imshow(x[j].permute(1, 2, 0).cpu())
            axes[0, j].axis('off')
            axes[1, j].imshow(y[j].permute(1, 2, 0).cpu())
            axes[1, j].axis('off')
            axes[2, j].imshow(y_fake[j].permute(1, 2, 0).cpu())
            axes[2, j].axis('off')
        plt.savefig(f'train_example_epoch_{epoch}.png')
    gen.train()

####################################################################################

def save_checkpoint(model, optimizer, filename):
    ''' Save checkpoint: saves model and optimizer state'''
    print("*** Saving checkpoint ***")
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, filename)

####################################################################################

def load_checkpoint(checkpoint_file_name, model, optimizer, learning_rate):
    ''' Load checkpoint: Load model and Load state, and set learning_rate '''
    if os.path.exists(checkpoint_file_name):
        print("*** Loading checkpoint ***")
        checkpoint = torch.load(checkpoint_file_name, map_location=DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate  # Load current learning rate
        
#####################################################################################


if __name__ == "__main__":
    print("Utils module loaded.")