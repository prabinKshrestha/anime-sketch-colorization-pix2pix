import torch
import torch.nn as nn

#####################################################################################

class DiscriminatorCNNBlock(nn.Module):
    """
    CNN Block module consists of a Convolutional Layer followed by Batch Normalization and Leaky ReLU activation.
    """
    def __init__(self, in_channels, out_channels, stride, padding, kernel_size=4, batch_normalize=True, leaky_relu=True):
        super().__init__()
        layers = []
        # Conv2d Layer
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False, padding_mode="reflect"))
        # BatchNorm if true
        if batch_normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        # LeakyReLU if true
        if leaky_relu:
            layers.append(nn.LeakyReLU(0.2))
        # Create Sequential Layers of Conv2d,BatchNorm and LeakyReLU
        self.conv_layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_layer(x)

####################################################################################
    
class Discriminator(nn.Module):
    """
    Discriminator model for image discrimination.
    """
    def __init__(self, in_channels=3):
        super().__init__()
        # Create Discriminator Model of pix2pix
        self.model = nn.Sequential(
            # [N, 64*2, 256, 256] => [N, 64, 128, 128]
            DiscriminatorCNNBlock(in_channels=in_channels*2,  out_channels=64, stride=2, padding=1, batch_normalize=False),
            # [N, 64, 128, 128] => [N, 128, 63, 63]
            DiscriminatorCNNBlock(in_channels=64,  out_channels=128, stride=2, padding=0),
            # [N, 128, 63, 63] => [N, 256, 30, 30]
            DiscriminatorCNNBlock(in_channels=128, out_channels=256, stride=2, padding=0),
            # [N, 256, 30, 30] => [N, 512, 27, 27]
            DiscriminatorCNNBlock(in_channels=256, out_channels=512, stride=1, padding=0),
            # [N, 512, 27, 27] => [N, 1, 26, 26]
            DiscriminatorCNNBlock(in_channels=512,  out_channels=1, stride=1, padding=1, batch_normalize=False, leaky_relu=False),
        )

    def forward(self, x, y):
        x = torch.cat([x,y], dim=1)
        return self.model(x)

####################################################################################

def discriminator_model_test():
    print("*** Testing Discriminator Model ***")
    x = torch.randn((5,3,256,256))
    y = torch.randn((5,3,256,256))
    model = Discriminator()
    preds = model(x,y)
    print(preds.shape) # should be [5, 1, 26, 26]


if __name__ == "__main__":
    discriminator_model_test()

    