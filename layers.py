#!/usr/bin/env python3.6

import torch.nn as nn
import torch.nn.functional as F


def convBatch(nin, nout, kernel_size=3, stride=1, padding=1, bias=False, layer=nn.Conv2d, dilation=1):
    return nn.Sequential(
        layer(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation),
        nn.BatchNorm2d(nout),
        nn.PReLU()
    )


def downSampleConv(nin, nout, kernel_size=3, stride=2, padding=1, bias=False):
    return nn.Sequential(
        convBatch(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
    )


class interpolate(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super().__init__()

        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, cin):
        return F.interpolate(cin, mode=self.mode, scale_factor=self.scale_factor)


def upSampleConv(nin, nout, kernel_size=3, upscale=2, padding=1, bias=False):
    return nn.Sequential(
        # nn.Upsample(scale_factor=upscale),
        interpolate(mode='nearest', scale_factor=upscale),
        convBatch(nin, nout, kernel_size=kernel_size, stride=1, padding=padding, bias=bias),
        convBatch(nout, nout, kernel_size=3, stride=1, padding=1, bias=bias),
    )


def conv_block(in_dim, out_dim, act_fn, kernel_size=3, stride=1, padding=1, dilation=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def conv_block_1(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=1),
        nn.BatchNorm2d(out_dim),
        nn.PReLU(),
    )
    return model


def conv_block_Asym(in_dim, out_dim, kernelSize):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=[kernelSize, 1], padding=tuple([2, 0])),
        nn.Conv2d(out_dim, out_dim, kernel_size=[1, kernelSize], padding=tuple([0, 2])),
        nn.BatchNorm2d(out_dim),
        nn.PReLU(),
    )
    return model


def conv_block_3_3(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_dim),
        nn.PReLU(),
    )
    return model


def conv_block_3(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        conv_block(in_dim, out_dim, act_fn),
        conv_block(out_dim, out_dim, act_fn),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
    )
    return model


def conv(nin, nout, kernel_size=3, stride=1, padding=1, bias=False, layer=nn.Conv2d,
         BN=False, ws=False, activ=nn.LeakyReLU(0.2), gainWS=2):
    convlayer = layer(nin, nout, kernel_size, stride=stride, padding=padding, bias=bias)
    layers = []
    if BN:
        layers.append(nn.BatchNorm2d(nout))
    if activ is not None:
        if activ == nn.PReLU:
            # to avoid sharing the same parameter, activ must be set to nn.PReLU (without '()')
            layers.append(activ(num_parameters=1))
        else:
            # if activ == nn.PReLU(), the parameter will be shared for the whole network !
            layers.append(activ)
    layers.insert(ws, convlayer)
    return nn.Sequential(*layers)


# TODO: Change order of block: BN + Activation + Conv
def conv_decod_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


# For UNet
class residualConv(nn.Module):
    def __init__(self, nin, nout):
        super(residualConv, self).__init__()
        self.convs = nn.Sequential(
            convBatch(nin, nout),
            nn.Conv2d(nout, nout, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nout)
        )
        self.res = nn.Sequential()
        if nin != nout:
            self.res = nn.Sequential(
                nn.Conv2d(nin, nout, kernel_size=1, bias=False),
                nn.BatchNorm2d(nout)
            )

    def forward(self, input):
        out = self.convs(input)
        return F.leaky_relu(out + self.res(input), 0.2)
