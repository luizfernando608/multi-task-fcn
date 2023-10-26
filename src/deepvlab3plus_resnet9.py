import torch
from torch import nn
import torch.nn.functional as F
import numpy as np



def conv_padding_same(kernel_size , **kwargs):
    
    padding = (kernel_size - 1) // 2

    return nn.Conv2d(**kwargs, padding = padding, kernel_size = kernel_size)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: bool,
    ) -> None:
        super(ResidualBlock, self).__init__()
        
        if input_channels != output_channels:
            raise ValueError("Invalid configuration: The number of input channels ({}) should match the number of output channels ({}).".format(input_channels, output_channels))

        
        self.downsample = downsample

        self.batch_norm1 = nn.BatchNorm2d(
            num_features=in_channels,
            momentum=0.9,
            eps=1e-05,
            affine=True,
            track_running_stats=True,
        )
        self.elu = nn.ELU()

        if downsample:
  
            self.conv_down1 = conv_padding_same(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2
            )

            self.conv_down2 = conv_padding_same(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2,
            )
            in_channels = out_channels


        self.conv1 = conv_padding_same(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
        )

        self.batch_norm2 = nn.BatchNorm2d(
            num_features=in_channels,
            momentum=0.9,
            eps=1e-05,
            affine=True,
            track_running_stats=True,
        )

        self.conv2 = conv_padding_same(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
        )

    def forward(self, x):
        x_init = x

        x = self.batch_norm1(x)
        x = self.elu(x)

        if self.downsample:
            x = self.conv_down1(x)
            x_init = self.conv_down2(x_init)

        x = self.batch_norm2(x)
        x = self.conv2(x)

        return x + x_init


class DeepLabv3Plus_resnet9(nn.Module):
    def __init__(self, num_ch, psize, num_class):
        super(DeepLabv3Plus_resnet9, self).__init__()
        
        self.num_ch_1 = num_ch
        self.psize = psize
        self.nb_class = num_class

    def enconder(self, x):
        pass

    def decoder_class(self, x):
        pass
    
    def decoder_aux(self, x):
        pass
    
    def forward(self, x):
        pass



if __name__ == "__main__":
    image = torch.randn(1, 64, 128, 128)

    model = ResidualBlock(in_channels=64, out_channels=64, downsample=False)

    with torch.no_grad():
        output = model(image)

    print(output.size())

    model = ResidualBlock(in_channels = 64, out_channels=64, downsample=True)

    with torch.no_grad():
        output = model(image)
    
    print(output.size())
    # model = DeepLabv3Plus_resnet9(10,
    #              num_ch_1 = 25,
    #              psize = 128,
    #              nb_class = 14)

    # model.eval()

    # image = torch.randn(1, 25, 128, 128)

    # with torch.no_grad():
    #     output_cl, output_depth = model.forward(image)
    # print(output_cl.size())
    # print(output_depth.size())
