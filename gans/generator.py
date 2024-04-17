import torch
import torch.nn as nn

#####################################################################################

class EncoderBlock(nn.Module):
    """
    Encoder block consists of convolutional layers followed by batch normalization and LeakyReLU activation.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Create Encoder conv layer
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv_layer(x)

#####################################################################################
    
class DecoderBlock(nn.Module):
    """
    Decoder block consists of transposed convolutional layers followed by batch normalization and ReLU activation.
    """
    def __init__(self, in_channels, out_channels, use_dropout=False):
        super().__init__()
        # Create decoder conv layer
        self.conv_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4,2,1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv_layer(x)
        return self.dropout(x) if self.use_dropout else x

#####################################################################################

class Generator(nn.Module):
    """
    Generator model consists of encoder and decoder blocks.
    """
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        ### Start Encoder Blocks ###
        self.encdr1 = nn.Sequential(
            nn.Conv2d(in_channels, features, 4,2,1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )                                                  # [N, 64, 128, 128]
        self.encdr2 = EncoderBlock(features, features*2)   # [N, 64*2=128, 64, 64]
        self.encdr3 = EncoderBlock(features*2, features*4) # [N, 64*4=256, 32, 32]
        self.encdr4 = EncoderBlock(features*4, features*8) # [N, 64*8=512, 16, 16] 
        self.encdr5 = EncoderBlock(features*8, features*8) # [N, 64*8=512, 8, 8]
        self.encdr6 = EncoderBlock(features*8, features*8) # [N, 64*8=512, 4, 4]
        self.encdr7 = EncoderBlock(features*8, features*8) # [N, 64*8=512, 2, 2]
        self.encdr8 = nn.Sequential(
            nn.Conv2d(features*8, features*8, 4,2,1, padding_mode="reflect"),
            nn.ReLU(),
        )                                                  # [N, 64*8=512, 1, 1] 
        ### Start Decoder Blocks with Skip Connection ###
        self.decdr8 = DecoderBlock(features*8, features*8, use_dropout=True)     # [N, 64*8=512, 2, 2] 
        self.decdr7 = DecoderBlock(features*8*2, features*8, use_dropout=True)   # [N, 64*8=512, 4, 4] 
        self.decdr6 = DecoderBlock(features*8*2, features*8, use_dropout=True)   # [N, 64*8=512, 8, 8] 
        self.decdr5 = DecoderBlock(features*8*2, features*8, use_dropout=False)  # [N, 64*8=512, 16, 16] 
        self.decdr4 = DecoderBlock(features*8*2, features*4, use_dropout=False)  # [N, 64*4=256, 32, 32] 
        self.decdr3 = DecoderBlock(features*4*2, features*2, use_dropout=False)  # [N, 64*2=128, 64, 64] 
        self.decdr2 = DecoderBlock(features*2*2, features, use_dropout=False)    # [N, 64, 128, 128] 
        self.decdr1 = nn.Sequential(
            nn.ConvTranspose2d(features*2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )                                                                        # [N, 3, 256, 256] 
        
    def forward(self, x):
        ## Start Encoder ###
        encdr1 = self.encdr1(x)                               # [N, 64, 128, 128]
        encdr2 = self.encdr2(encdr1)                          # [N, 64*2=128, 64, 64]
        encdr3 = self.encdr3(encdr2)                          # [N, 64*4=256, 32, 32]
        encdr4 = self.encdr4(encdr3)                          # [N, 64*8=512, 16, 16]
        encdr5 = self.encdr5(encdr4)                          # [N, 64*8=512, 8, 8]
        encdr6 = self.encdr6(encdr5)                          # [N, 64*8=512, 4, 4]
        encdr7 = self.encdr7(encdr6)                          # [N, 64*8=512, 2, 2]
        encdr8 = self.encdr8(encdr7)                          # [N, 64*8=512, 1, 1]
        ## Start Decoder with Skip Connection ###
        decdr8 = self.decdr8(encdr8)                          # [N, 64*8=512, 2, 2]   
        decdr7 = self.decdr7(torch.cat([decdr8, encdr7], 1))  # [N, 64*8=512, 4, 4] 
        decdr6 = self.decdr6(torch.cat([decdr7, encdr6], 1))  # [N, 64*8=512, 8, 8] 
        decdr5 = self.decdr5(torch.cat([decdr6, encdr5], 1))  # [N, 64*8=512, 16, 16]
        decdr4 = self.decdr4(torch.cat([decdr5, encdr4], 1))  # [N, 64*4=256, 32, 32]
        decdr3 = self.decdr3(torch.cat([decdr4, encdr3], 1))  # [N, 64*2=128, 64, 64]
        decdr2 = self.decdr2(torch.cat([decdr3, encdr2], 1))  # [N, 64, 128, 128] 
        decdr1 = self.decdr1(torch.cat([decdr2, encdr1], 1))  # [N, 3, 256, 256] 
        return decdr1


#####################################################################################

def generetor_model_test():
    print("*** Testing Generator Model ***")
    x = torch.randn((5, 3, 256, 256))
    model = Generator(in_channels=3, features=64)
    preds = model(x) 
    print(preds.shape)  # expect [5, 3, 256, 256] => generate 3 channel colored image

#####################################################################################

if __name__ == "__main__":
    generetor_model_test()