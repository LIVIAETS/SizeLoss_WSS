#!/usr/bin/env python3.6

import os
from pathlib import Path

import torch
import numpy as np
import torchvision
import torch.nn as nn
from torch.autograd import Variable


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100,
                     fill='=', empty=' ', tip='>', begin='[', end=']', done="[DONE]", clear=True):
    """
    Print iterations progress.
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required : current iteration                          [int]
        total       - Required : total iterations                           [int]
        prefix      - Optional : prefix string                              [str]
        suffix      - Optional : suffix string                              [str]
        decimals    - Optional : positive number of decimals in percent     [int]
        length      - Optional : character length of bar                    [int]
        fill        - Optional : bar fill character                         [str] (ex: 'â– ', 'â–ˆ', '#', '=')
        empty       - Optional : not filled bar character                   [str] (ex: '-', ' ', 'â€¢')
        tip         - Optional : character at the end of the fill bar       [str] (ex: '>', '')
        begin       - Optional : starting bar character                     [str] (ex: '|', 'â–•', '[')
        end         - Optional : ending bar character                       [str] (ex: '|', 'â–', ']')
        done        - Optional : display message when 100% is reached       [str] (ex: "[DONE]")
        clear       - Optional : display completion message or leave as is  [str]
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength
    if iteration != total:
        bar = bar + tip
    bar = bar + empty * (length - filledLength - len(tip))
    display = '\r{prefix}{begin}{bar}{end} {percent}%{suffix}' \
              .format(prefix=prefix, begin=begin, bar=bar, end=end, percent=percent, suffix=suffix)
    print(display, end=''),   # comma after print() required for python 2
    if iteration == total:      # print with newline on complete
        if clear:               # display given complete message with spaces to 'erase' previous progress bar
            finish = '\r{prefix}{done}'.format(prefix=prefix, done=done)
            if hasattr(str, 'decode'):   # handle python 2 non-unicode strings for proper length measure
                finish = finish.decode('utf-8')
                display = display.decode('utf-8')
            clear = ' ' * max(len(display) - len(finish), 0)
            print(finish + clear)
        else:
            print('')


class computeDiceOneHotBinary(nn.Module):
    def __init__(self):
        super(computeDiceOneHotBinary, self).__init__()

    def dice(self, input, target):
        inter = (input * target).float().sum()
        sum = input.sum() + target.sum()
        if (sum == 0).all():
            return (2 * inter + 1e-8) / (sum + 1e-8)

        return 2 * (input * target).float().sum() / (input.sum() + target.sum())

    def inter(self, input, target):
        return (input * target).float().sum()

    def sum(self, input, target):
        return input.sum() + target.sum()

    def forward(self, pred, GT):
        # GT is 4x320x320 of 0 and 1
        # pred is converted to 0 and 1
        batchsize = GT.size(0)
        DiceN = to_var(torch.zeros(batchsize, 2))
        DiceB = to_var(torch.zeros(batchsize, 2))

        for i in range(batchsize):
            DiceN[i, 0] = self.inter(pred[i, 0], GT[i, 0])
            DiceB[i, 0] = self.inter(pred[i, 1], GT[i, 1])

            DiceN[i, 1] = self.sum(pred[i, 0], GT[i, 0])
            DiceB[i, 1] = self.sum(pred[i, 1], GT[i, 1])

        return DiceN, DiceB


def DicesToDice(Dices):
    # print('dtd')
    # print(Dices.data.cpu().numpy())
    sums = Dices.sum(dim=0)
    # print(sums.data.cpu().numpy())
    return (2 * sums[0] + 1e-8) / (sums[1] + 1e-8)


def getSingleImageBin(pred):
    # input is a 4-channels image corresponding to the predictions of the net
    # output is a gray level image (1 channel) of the segmentation with "discrete" values
    Val = to_var(torch.zeros(2))
    Val[1] = 1.0

    x = predToSegmentation(pred)

    out = x * Val.view(1, 2, 1, 1)
    return out.sum(dim=1, keepdim=True)


def predToSegmentation(pred):
    Max = pred.max(dim=1, keepdim=True)[0]
    x = pred / Max
    return (x == 1).float()


def getOneHotSegmentation_Bin(batch):
    backgroundVal = 0
    label1 = 1.0

    oneHotLabels = torch.cat((batch == backgroundVal, batch == label1),  dim=1)
    return oneHotLabels.float()


def inference(net, temperature, img_batch, batch_size, epoch, deepSupervision, modelName, minSize, maxSize):
    # directory = 'Results/ImagesViolationConstraint/NIPS/' + modelName + '/Epoch_' + str(epoch)
    directory = str(Path("Results", "ImagesViolationConstraint", "MIDL", modelName, f"Epoch_{epoch}"))
    if not os.path.exists(directory):
        os.makedirs(directory)

    total = len(img_batch)

    Dice1 = torch.zeros(total, 2)

    net.eval()

    img_names_ALL = []

    softMax = nn.Softmax()
    softMax.cuda()

    dice = computeDiceOneHotBinary().cuda()

    sizesGT = []
    sizesPred = []

    for i, data in enumerate(img_batch):
        printProgressBar(i, total, prefix="[Inference] Getting segmentations...", length=30)
        image, labels, labels_weak, img_names = data
        img_names_ALL.append(img_names[0].split('/')[-1].split('.')[0])

        MRI = to_var(image)
        Segmentation = to_var(labels)

        if not deepSupervision:
            segmentation_prediction = net(MRI)
        else:
            segmentation_prediction, seg3, seg2, seg1 = net(MRI)

        Segmentation_planes = getOneHotSegmentation_Bin(Segmentation)

        pred_y = softMax(segmentation_prediction/temperature)

        softmax_y = pred_y.cpu().data.numpy()
        softB = softmax_y[:, 1, :, :]

        # Hard-Size
        pixelsClassB = np.where(softB > 0.5)

        predLabTemp = np.zeros(softB.shape)
        predLabTemp[pixelsClassB] = 1.0
        sizePredNumpy = predLabTemp.sum()
        # minSize = 97.9
        # maxSize = 1722.6

        idx = np.where(labels.numpy() == 1.0)
        sizeLV_GT = len(idx[0])
        sizesGT.append(sizeLV_GT)
        sizesPred.append(sizePredNumpy)

        if sizeLV_GT > 0:
            if sizePredNumpy < minSize:
                out = torch.cat((MRI, pred_y[:, 1, :, :].view(1, 1, 256, 256), Segmentation))
                name2save = img_names[0].split('./ACDC-2D-All/val/Img/')
                name2save = name2save[1].split('.png')
                torchvision.utils.save_image(out.data, os.path.join(directory, name2save[0]+'_Lower_'+str(minSize-sizePredNumpy)+'.png'),
                                             nrow=batch_size,
                                             padding=2,
                                             normalize=False,
                                             range=None,
                                             scale_each=False,
                                             pad_value=0)

            if sizePredNumpy > maxSize:
                out = torch.cat((MRI, pred_y[:, 1, :, :].view(1, 1, 256, 256), Segmentation))
                name2save = img_names[0].split('./ACDC-2D-All/val/Img/')
                name2save = name2save[1].split('.png')
                torchvision.utils.save_image(out.data, os.path.join(directory, name2save[0]+'_Upper_'+str(sizePredNumpy-maxSize)+'.png'),
                                             nrow=batch_size,
                                             padding=2,
                                             normalize=False,
                                             range=None,
                                             scale_each=False,
                                             pad_value=0)

        else:
            if sizePredNumpy > 0:
                out = torch.cat((MRI, pred_y[:, 1, :, :].view(1, 1, 256, 256), Segmentation))
                name2save = img_names[0].split('./ACDC-2D-All/val/Img/')
                name2save = name2save[1].split('.png')
                torchvision.utils.save_image(out.data, os.path.join(directory, name2save[0]+'_'+str(sizePredNumpy)+'.png'),
                                             nrow=batch_size,
                                             padding=2,
                                             normalize=False,
                                             range=None,
                                             scale_each=False,
                                             pad_value=0)

        DicesN, Dices1 = dice(pred_y, Segmentation_planes)
        Dice1[i] = Dices1.data
    printProgressBar(total, total, done="[Inference] Segmentation Done !")

    ValDice1 = DicesToDice(Dice1)

    return [ValDice1, sizesGT, sizesPred]
