import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


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
    model = DeepLabv3Plus_resnet9(10, 
                 num_ch_1=25,
                 psize = 128,
                 nb_class = 14)
    
    model.eval()

    image = torch.randn(1, 25, 128, 128)
    
    with torch.no_grad():
        output_cl, output_depth = model.forward(image)
    print(output_cl.size())
    print(output_depth.size())