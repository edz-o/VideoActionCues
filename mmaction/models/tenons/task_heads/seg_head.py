import torch
import torch.nn as nn
import torch.nn.functional as F
from ...registry import HEADS

class up(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor=(1,2,2), bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        else:
            raise Warning('Not fully implemented')
            self.up = nn.ConvTranspose3d(in_ch//2, in_ch//2, 3, stride=(1, 2, 2))

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        #x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
        #diffY // 2, diffY - diffY//2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
        nn.Conv3d(in_ch, out_ch, (1,3,3), padding=(0,1,1)),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, (1,3,3), padding=(0,1,1)),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True)

        )

    def forward(self, x):
        x = self.conv(x)
        return x


@HEADS.register_module
class SegHead(nn.Module):

    def __init__(self, n_classes, input_size):

        super(SegHead, self).__init__()
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = input_size
        self.up1 = up(3072, 256, scale_factor=(1,2,2))
        self.up2 = up(768, 64, scale_factor=(1,2,2))
        self.up3 = up(320, 64, scale_factor=(2,2,2))
        self.out_conv = nn.Conv3d(64, n_classes, kernel_size=1, stride=1)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, feat, seg=None):
        #assert input_size is not None
        #size = (seg.shape[3], seg.shape[4])
        size = self.input_size
        x = self.up1(feat[3], feat[2])
        x = self.up2(x, feat[1])
        x = self.up3(x, feat[0])
        x = self.out_conv(x)
        x = nn.Upsample((x.shape[2],)+size, mode='nearest')(x)

        if seg is not None:
            seg = F.interpolate(seg.double(), x.shape[2:], mode='nearest')
            seg = seg.squeeze(1).long()

            loss = self.loss_func(x, seg)
            return x, loss
        else:
            return x




