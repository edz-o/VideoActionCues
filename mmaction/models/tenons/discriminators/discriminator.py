import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import functools

from ...registry import DISCRIMINATORS

@DISCRIMINATORS.register_module
class FCDiscriminator(nn.Module):

    def __init__(self, num_classes, ndf = 64):
        super(FCDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.bce_loss = nn.BCEWithLogitsLoss()


    def forward(self, x, lbl):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        self.loss = self.bce_loss(x, Variable(torch.FloatTensor(x.data.size()).fill_(lbl)).cuda())

        return x

    def adjust_learning_rate(self, args, optimizer, i):
        if args.model == 'DeepLab':
            lr = args.learning_rate_D * ((1 - float(i) / args.num_steps) ** (args.power))
            optimizer.param_groups[0]['lr'] = lr
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]['lr'] = lr * 10
        else:
            optimizer.param_groups[0]['lr'] = args.learning_rate_D * (0.1**(int(i/50000)))
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]['lr'] = args.learning_rate_D * (0.1**(int(i/50000))) * 2


@DISCRIMINATORS.register_module
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, lambda_adv_1=0.001, norm_layer=nn.BatchNorm3d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        self.lambda_adv_1 = lambda_adv_1
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 1
        padw = 0
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, input):
        x = self.model(input)
        return x

    def loss(self, x, lbl):
        loss = self.bce_loss(x, torch.FloatTensor(x.data.size()).fill_(lbl).cuda())
        return torch.unsqueeze(loss, 0)

    def freeze(self, freeze=True):
        for m in self.modules():
            #print(type(m))
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.BatchNorm3d):
                for p in m.parameters():
                    p.requires_grad = not freeze


@DISCRIMINATORS.register_module
class NLayerDiscriminator2D(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator2D, self).__init__()
        self.loss = None
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 1
        padw = 0
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, input):
        x = self.model(input)
        return x

    def loss(self, x, lbl):
        self.loss = self.bce_loss(x, torch.FloatTensor(x.data.size()).fill_(lbl).cuda())
        return torch.unsqueeze(self.loss, 0)
