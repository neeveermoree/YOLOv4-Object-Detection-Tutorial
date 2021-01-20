import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class DarkNet53(nn.Module):
    def __init__(self):
        super(DarkNet53, self).__init__()
        kernel_1 = (1, 1)
        kernel_2 = (2, 2)
        kernel_3 = (3, 3)
        # in -> out        
        self.conv1 = nn.Conv2d(3, 32, kernel_3, stride=1, padding=1)   # 256 -> 256
        self.conv2 = nn.Conv2d(32, 64, kernel_3, stride=2, padding=1)  # 256 -> 128 

        # first residual block
        self.conv3 = nn.Conv2d(64, 32, kernel_1, stride=1)             # 128 -> 128
        self.conv4 = nn.Conv2d(32, 64, kernel_3, padding=1)            # 128 -> 128


dn = DarkNet53()
