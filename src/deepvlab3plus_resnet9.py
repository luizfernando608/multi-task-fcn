import torch
from torch import nn
import torch.nn.functional as F
import numpy as np



def conv_padding_same(kernel_size , **kwargs):
    if "dilation" in kwargs.keys():
        padding = ((kernel_size - 1) * kwargs["dilation"]) // 2

    else:
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
                in_channels=in_channels,
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


class ASPPModule(nn.Module):
    def __init__(self, pool_height, pool_width, in_channels, out_channels = 128):
        super(ASPPModule, self).__init__()
        

        self.aspp1x1 = conv_padding_same(in_channels = in_channels, out_channels=out_channels, kernel_size=1)
        self.aspp3x3_1 = conv_padding_same(in_channels = in_channels, out_channels=out_channels, kernel_size=3, dilation=6)
        self.aspp3x3_2 = conv_padding_same(in_channels = in_channels, out_channels=out_channels, kernel_size=3, dilation=12)
        self.aspp3x3_3 = conv_padding_same(in_channels = in_channels, out_channels=out_channels, kernel_size=3, dilation=18)

        self.image_feature = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            conv_padding_same(in_channels = in_channels, out_channels=out_channels, kernel_size=1),
            nn.Upsample(size=(pool_height, pool_width), mode='bilinear', align_corners=True)
        )

        self.batch_norm = nn.BatchNorm2d(out_channels*5, 
                                         momentum=0.9,
                                            eps=1e-05,
                                            affine=True,
                                            track_running_stats=True,)
        
        self.relu = nn.ELU()


    def forward(self, x):

        aspp1x1 = self.aspp1x1(x)
        aspp3x3_1 = self.aspp3x3_1(x)
        aspp3x3_2 = self.aspp3x3_2(x)
        aspp3x3_3 = self.aspp3x3_3(x)

        image_feature = self.image_feature(x)

        outputs = torch.cat([aspp1x1, aspp3x3_1, aspp3x3_2, aspp3x3_3, image_feature], dim=1)
        outputs = self.batch_norm(outputs)
        outputs = self.relu(outputs)

        return outputs



class DeepLabv3Plus_resnet9(nn.Module):
    def __init__(self, num_ch, psize, num_class):
        super(DeepLabv3Plus_resnet9, self).__init__()

        self.num_ch = num_ch
        self.psize = psize
        self.nb_class = num_class

        self.conv1 = conv_padding_same(in_channels = self.num_ch,
                                       out_channels = 64,
                                       kernel_size = 3, 
                                       stride = 1)
        
        self.res_block1 = ResidualBlock(in_channels=64,
                                        out_channels=64,
                                        downsample=False)
        
        self.res_block2 = ResidualBlock(in_channels=64,
                                        out_channels=128,
                                        downsample=True)
        
        self.res_block3 = ResidualBlock(in_channels=128,
                                        out_channels=128,
                                        downsample=False)
    
        self.res_block3 = ResidualBlock(in_channels=128,
                                        out_channels=256,
                                        downsample=True)
        
        self.res_block4 = ResidualBlock(in_channels=256,
                                        out_channels=256,
                                        downsample=False)

        self.batch_norm1 = nn.BatchNorm2d(num_features=256, 
                                          momentum=0.9, 
                                          eps=1e-05, 
                                          affine=True, 
                                          track_running_stats=True)

        self.elu1 = nn.ELU()

        self.atrous_pyramid_pooling = ASPPModule(pool_height=32, 
                                                 pool_width=32, 
                                                 in_channels=256, 
                                                 out_channels=128)



    def enconder(self, x):
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        x_skip = x

        x = self.res_block4(x)
        x = self.batch_norm1(x)
        x = self.elu1(x)
                
        return x, x_skip
    
    def decoder_class(self, x, x_skip):
        self.atrous_pyramid_pooling(x)
        return x

    def decoder_aux(self, x, x_skip):
        pass

    def forward(self, x):
        x, x_skip = self.enconder(x)

        x = self.decoder_class(x, x_skip)

        return x



if __name__ == "__main__":

    image = torch.randn(1, 25, 128, 128)

    model = DeepLabv3Plus_resnet9(
        num_ch = 25,
        num_class = 14,
        psize = 128,
    )

    output = model(image)
    print(output.size)

    image = torch.randn(1, 128, 32, 32)

    model = ASPPModule(image.size(2), 
                       image.size(3), 
                       image.size(1), 
                       128)
    model.eval()
    
    with torch.no_grad():
        output = model(image)

    # image = torch.randn(1, 25, 128, 128)

    # model = ResidualBlock(in_channels=64, out_channels=64, downsample=False)

    # with torch.no_grad():
    #     output = model(image)

    # print(output.size())

    # model = ResidualBlock(in_channels = 64, out_channels=64, downsample=True)

    # with torch.no_grad():
    #     output = model(image)
    
    # print(output.size())

    

    model.eval()

    image = torch.randn(1, 25, 128, 128)

    with torch.no_grad():
        x, x_skip = model.enconder(image)
    
    print(x.size())
    print(x.size())
