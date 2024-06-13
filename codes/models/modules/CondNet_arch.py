import functools
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil
import models.modules.sft_arch as sft
from .unet_parts import *


# class CondNet(nn.Module):
#     ''' modified SRResNet'''
#
#     def __init__(self, n_channels, n_classes, bilinear=False):
#         super(CondNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#
#         factor = 2 if bilinear else 1
#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         self.down4 = Down(512, 1024 // factor)
#
#         self.up1 = Up(1024, 512 // factor, bilinear)
#         self.up2 = Up(512, 256 // factor, bilinear)
#         self.up3 = Up(256, 128 // factor, bilinear)
#         self.up4 = Up(128, 64, bilinear)
#         self.outc = OutConv(64, n_classes)

# # Unet
#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         # x5 = self.down4(x4)
#         # x = self.up1(x5, x4)
#         x = self.up2(x4, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits

class CondNet(nn.Module):
    ''' modified SRResNet'''

    def __init__(self, n_channels, n_classes, bilinear=False):
        super(CondNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    # # HWNet0
    # def forward(self, x):
    #     x1 = self.inc(x)
    #     x2 = self.down1(x1)
    #     x3 = self.down2(x2)
    #     x4 = self.down3(x3)
    #     x = self.up1(x4, x3)
    #     x = self.up2(x, x2)
    #     x = self.up3(x, x1)
    #     logits = self.outc(x)
    #     return logits

    # # HWNet1
    # def forward(self, x):
    #     x1 = self.inc(x)
    #     x2 = self.down1(x1)
    #     x3 = self.down2(x2)
    #     x4 = self.down3(x3)
    #     x5 = self.up1(x4, x3)
    #     x = self.down3(x5)
    #     x = self.up1(x, x5)
    #     x = self.up2(x, x2)
    #     x = self.up3(x, x1)
    #     logits = self.outc(x)
    #     return logits

    # HWNet2
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.up1(x4, x3)
        x6 = self.up2(x5, x2)
        x7 = self.down2(x6)
        x8 = self.down3(x7)
        x9 = self.up1(x8, x7)
        x10 = self.up2(x9, x6)
        x11 = self.up3(x10, x1)
        logits = self.outc(x11)
        return logits

    # # HWNet3
    # def forward(self, x):
    #     x1 = self.inc(x)
    #     x2 = self.down1(x1)
    #     x3 = self.down2(x2)
    #     x4 = self.down3(x3)
    #     x5 = self.up1(x4, x3)
    #     x6 = self.up2(x5, x2)
    #     x7 = self.up3(x6, x1)
    #     x8 = self.down1(x7)
    #     x9 = self.down2(x8)
    #     x10 = self.down3(x9)
    #     x11 = self.up1(x10, x9)
    #     x12 = self.up2(x11, x8)
    #     x13 = self.up3(x12, x7)
    #     logits = self.outc(x13)
    #     return logits



# class CondNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=False):
#         super(CondNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#
#         factor = 2 if bilinear else 1
#         self.inc = (DoubleConv(n_channels, 64))
#         self.down1 = (Down(64, 128))
#         self.down2 = (Down(128, 256))
#         self.down3 = (Down(256, 512))
#         self.down4 = (Down(256, 512))
#         self.up1 = (Up(512, 256 // factor, bilinear))
#         self.up2 = (Up(512, 256 // factor, bilinear))
#         self.up3 = (Up(256, 128 // factor, bilinear))
#         self.up4 = (Up(128, 64, bilinear))
#         self.outc = (OutConv(64, n_classes))
#
#     # HWNet1(independent)
#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.up1(x4, x3)
#         x6 = self.down4(x5)
#         x7 = self.up2(x6, x5)
#         x8 = self.up3(x7, x2)
#         x9 = self.up4(x8, x1)
#         logits = self.outc(x9)
#         return logits

# class CondNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=False):
#         super(CondNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#
#         factor = 2 if bilinear else 1
#         self.inc = (DoubleConv(n_channels, 64))
#         self.down1 = (Down(64, 128))
#         self.down2 = (Down(128, 256))
#         self.down3 = (Down(256, 512))
#         self.down4 = (Down(128, 256))
#         self.down5 = (Down(256, 512))
#         self.up1 = (Up(512, 256 // factor, bilinear))
#         self.up2 = (Up(256, 128 // factor, bilinear))
#         self.up3 = (Up(128, 64, bilinear))
#         self.up4 = (Up(512, 256 // factor, bilinear))
#         self.up5 = (Up(256, 128 // factor, bilinear))
#         self.outc = (OutConv(64, n_classes))
#
#     # HWNet2(independent)
#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.up4(x4, x3)
#         x6 = self.up5(x5, x2)
#         x7 = self.down4(x6)
#         x8 = self.down5(x7)
#         x9 = self.up1(x8, x7)
#         x10 = self.up2(x9, x6)
#         x11 = self.up3(x10, x1)
#         logits = self.outc(x11)
#         return logits
#
# class CondNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=False):
#         super(CondNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#
#         factor = 2 if bilinear else 1
#         self.inc = (DoubleConv(n_channels, 64))
#         self.down1 = (Down(64, 128))
#         self.down2 = (Down(128, 256))
#         self.down3 = (Down(256, 512))
#         self.down4 = (Down(64, 128))
#         self.down5 = (Down(128, 256))
#         self.down6 = (Down(256, 512))
#         self.up1 = (Up(512, 256 // factor, bilinear))
#         self.up2 = (Up(256, 128 // factor, bilinear))
#         self.up3 = (Up(128, 64, bilinear))
#         self.up4 = (Up(512, 256 // factor, bilinear))
#         self.up5 = (Up(256, 128 // factor, bilinear))
#         self.up6 = (Up(128, 64, bilinear))
#         self.outc = (OutConv(64, n_classes))
#
#     # HWNet3(independent)
#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.up1(x4, x3)
#         x6 = self.up2(x5, x2)
#         x7 = self.up3(x6, x1)
#         x8 = self.down4(x7)
#         x9 = self.down5(x8)
#         x10 = self.down6(x9)
#         x11 = self.up4(x10, x9)
#         x12 = self.up5(x11, x8)
#         x13 = self.up6(x12, x7)
#         logits = self.outc(x13)
#         return logits


