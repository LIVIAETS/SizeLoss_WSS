#!/usr/bin/env python3.6

import os

import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

import medicalDataLoader
from ENet import ENet
from utils import to_var
from utils import computeDiceOneHotBinary, predToSegmentation, inference, DicesToDice, printProgressBar
from losses import Partial_CE, MIL_Loss, Size_Loss


def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def getOneHot_Encoded_Segmentation(batch):
    backgroundVal = 0
    foregroundVal = 1.0
    # pdb.set_trace()
    oneHotLabels = torch.cat((batch == backgroundVal, batch == foregroundVal), dim=1)
    return oneHotLabels.float()


def runTraining():
    print('-' * 40)
    print('~~~~~~~~  Starting the training... ~~~~~~')
    print('-' * 40)

    # Batch size for training MUST be 1 in weakly/semi supervised learning if we want to impose constraints.
    batch_size = 1
    batch_size_val = 1
    lr = 0.0005
    epoch = 1000

    root_dir = './ACDC-2D-All'
    model_dir = 'model'

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set = medicalDataLoader.MedicalImageDataset('train',
                                                      root_dir,
                                                      transform=transform,
                                                      mask_transform=mask_transform,
                                                      augment=False,
                                                      equalize=False)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              num_workers=5,
                              shuffle=False)

    val_set = medicalDataLoader.MedicalImageDataset('val',
                                                    root_dir,
                                                    transform=transform,
                                                    mask_transform=mask_transform,
                                                    equalize=False)

    val_loader = DataLoader(val_set,
                            batch_size=batch_size_val,
                            num_workers=5,
                            shuffle=False)

    minVal = 97.9
    maxVal = 1722.6
    minSize = torch.FloatTensor(1)
    minSize.fill_(np.int64(minVal).item())
    maxSize = torch.FloatTensor(1)
    maxSize.fill_(np.int64(maxVal).item())

    print("~~~~~~~~~~~ Creating the model ~~~~~~~~~~")
    num_classes = 2

    netG = ENet(1, num_classes)

    netG.apply(weights_init)
    softMax = nn.Softmax()
    Dice_loss = computeDiceOneHotBinary()

    modelName = 'WeaklySupervised_CE-2_b'

    print(' Model name: {}'.format(modelName))
    partial_ce = Partial_CE()
    mil_loss = MIL_Loss()
    size_loss = Size_Loss()

    if torch.cuda.is_available():
        netG.cuda()
        softMax.cuda()
        Dice_loss.cuda()

    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

    BestDice, BestEpoch = 0, 0

    dBAll = []
    Losses = []

    annotatedPixels = 0
    totalPixels = 0

    print(" ~~~~~~~~~~~ Starting the training ~~~~~~~~~~")
    print(' --------- Params: ---------')
    print(' - Lower bound: {}'.format(minVal))
    print(' - Upper bound: {}'.format(maxVal))
    for i in range(epoch):
        netG.train()
        lossVal = []
        lossVal1 = []

        totalImages = len(train_loader)
        for j, data in enumerate(train_loader):
            image, labels, weak_labels, img_names = data

            # prevent batchnorm error for batch of size 1
            if image.size(0) != batch_size:
                continue

            optimizerG.zero_grad()
            netG.zero_grad()

            MRI = to_var(image)
            Segmentation = to_var(labels)
            weakAnnotations = to_var(weak_labels)

            segmentation_prediction = netG(MRI)

            annotatedPixels = annotatedPixels + weak_labels.sum()
            totalPixels = totalPixels + weak_labels.shape[2]*weak_labels.shape[3]
            temperature = 0.1
            predClass_y = softMax(segmentation_prediction/temperature)
            Segmentation_planes = getOneHot_Encoded_Segmentation(Segmentation)
            segmentation_prediction_ones = predToSegmentation(predClass_y)

            # lossCE_numpy = partial_ce(segmentation_prediction, Segmentation_planes, weakAnnotations)
            lossCE_numpy = partial_ce(predClass_y, Segmentation_planes, weakAnnotations)

            # sizeLoss_val = size_loss(segmentation_prediction, Segmentation_planes, Variable(minSize), Variable(maxSize))
            sizeLoss_val = size_loss(predClass_y, Segmentation_planes, Variable(minSize), Variable(maxSize))

            # MIL_Loss_val = mil_loss(predClass_y, Segmentation_planes)

            # Dice loss (ONLY USED TO COMPUTE THE DICE. This DICE loss version does not work)
            DicesN, DicesB = Dice_loss(segmentation_prediction_ones, Segmentation_planes)
            DiceN = DicesToDice(DicesN)
            DiceB = DicesToDice(DicesB)

            Dice_score = (DiceB + DiceN) / 2

            # Choose between the different models
            # lossG = lossCE_numpy + MIL_Loss_val
            lossG = lossCE_numpy + sizeLoss_val
            # lossG = lossCE_numpy
            # lossG = sizeLoss_val

            lossG.backward(retain_graph=True)
            optimizerG.step()

            lossVal.append(lossG.data[0])
            lossVal1.append(lossCE_numpy.data[0])

            printProgressBar(j + 1, totalImages,
                             prefix="[Training] Epoch: {} ".format(i),
                             length=15,
                             suffix=" Mean Dice: {:.4f}, Dice1: {:.4f} ".format(
                                 Dice_score.data[0],
                                 DiceB.data[0]))

        deepSupervision = False
        printProgressBar(totalImages, totalImages,
                         done=f"[Training] Epoch: {i}, LossG: {np.mean(lossVal):.4f}, lossMSE: {np.mean(lossVal1):.4f}")

        Losses.append(np.mean(lossVal))
        d1, sizeGT, sizePred = inference(netG, temperature, val_loader, batch_size, i, deepSupervision, modelName,
                                         minVal, maxVal)

        dBAll.append(d1)

        directory = 'Results/Statistics/MIDL/' + modelName
        if not os.path.exists(directory):
            os.makedirs(directory)

        np.save(os.path.join(directory, modelName + '_Losses.npy'), Losses)
        np.save(os.path.join(directory, modelName + '_dBAll.npy'), dBAll)

        currentDice = d1

        print(" [VAL] DSC: (1): {:.4f} ".format(d1))
        # saveImagesSegmentation(netG, val_loader_save_imagesPng, batch_size_val_savePng, i, 'test', False)

        if currentDice > BestDice:
            BestDice = currentDice
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save(netG, os.path.join(model_dir, "Best_" + modelName + ".pkl"))

        if i % (BestEpoch + 10):
            for param_group in optimizerG.param_groups:
                param_group['lr'] = lr


if __name__ == '__main__':
    runTraining()
