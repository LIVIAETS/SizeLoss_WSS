#!/usr/env/bin python3.6

import pdb

import torch
import numpy as np


class Partial_CE(torch.autograd.Function):
    def forward(self, input, target, weakLabels):
        self.save_for_backward(input, target, weakLabels)
        # b, c, w, h = input.shape
        # assert target.shape == input.shape
        # assert weakLabels.shape == (b, 1, w, h)

        # assert np.allclose(input[:, 0, ...].cpu().numpy(), 1 - input[:, 1, ...].cpu().numpy(), atol=1e-2)

        eps = 1e-20

        softmax_y = input.cpu().numpy()
        numPixelsNonMasked = weakLabels.sum()

        # Mask the non-annotated pixels
        if (numPixelsNonMasked > 0):
            loss = - np.sum(np.log(softmax_y[:, 1, :, :]+eps)*(weakLabels.view(1, 256, 256)).cpu().numpy())
            loss /= numPixelsNonMasked
        else:
            loss = 0.0

        lossT = torch.FloatTensor(1)
        lossT.fill_(np.float32(loss).item())

        return lossT.cuda()   # a single number (averaged loss over batch samples)

    def backward(self, grad_output):
        input, target, weakLabels = self.saved_variables
        eps = 1e-10

        oneHotLabels = torch.cat((weakLabels == 0, weakLabels == 1), dim=0).view(input.shape).float()
        numPixelsNonMasked = weakLabels.sum().data[0]

        # softmax_y = input.cpu().data.numpy()
        softmax_y = input.data

        # Mask the predictions to only annotated pixels
        mask = oneHotLabels
        mask[:, 0, :, :] = 0

        if numPixelsNonMasked > 0:
            grad_input = -oneHotLabels/(softmax_y+eps)
            grad_input *= mask
            grad_input /= numPixelsNonMasked
        else:
            grad_input = torch.FloatTensor(1)
            grad_input.fill_(0.0)

        return grad_input.cuda(), None, None


class MIL_Loss(torch.autograd.Function):
    def forward(self, input, target):
        self.save_for_backward(input, target)

        softmax_y = input.cpu().numpy()
        softB = softmax_y[:, 1, :, :]

        sizePredNumpy = softB.sum()
        sizePred = torch.FloatTensor(1)
        sizePred.fill_(sizePredNumpy.item())

        # Let's use the target (annotation) to know whether there some exist some target or not
        if target[:, 1, :, :].sum() > 0:
            loss = ((sizePred - 1)**2)/(softB.shape[1]*softB.shape[2])
        else:
            loss = ((sizePred)**2)/(softB.shape[1]*softB.shape[2])

        lossT = loss/100

        if np.isnan(loss.numpy()[0]):
            pdb.set_trace()

        return lossT.cuda()   # a single number (averaged loss over batch samples)

    def backward(self, grad_output):
        input, target = self.saved_variables

        m = input.shape[2]*input.shape[3]

        # Compute the hard size of the prediction
        softmax_y = input.cpu().data.numpy()
        softB = softmax_y[:, 1, :, :]

        # Soft Dice
        sizePred = softB.sum()
        sizePredNumpy = softB.sum()
        sizePred = torch.FloatTensor(1)
        sizePred.fill_(sizePredNumpy.item())

        # TO-DO. Currently, the loss is weighted by a hard-coded value (100). Add this as input parameter
        if target[:, 1, :, :].sum().cpu().data.numpy() > 0:
            lossValue = 2 * (sizePred-1)/(100*m)
        else:
            lossValue = 2 * (-sizePred)/(100*m)

        grad_inputA = np.zeros((softmax_y.shape[0], 1, softmax_y.shape[2], softmax_y.shape[3]), dtype='float32')
        grad_inputB = np.zeros((softmax_y.shape[0], 1, softmax_y.shape[2], softmax_y.shape[3]), dtype='float32')

        grad_inputB.fill(lossValue.numpy()[0])

        grad_input = np.concatenate((grad_inputA, grad_inputB), 1)

        return torch.Tensor(grad_input).cuda(), None


class Size_Loss(torch.autograd.Function):
    def forward(self, input, target, lower_B, upper_B):
        self.save_for_backward(input, target, lower_B, upper_B)

        # Compute the hard size of the prediction
        softmax_y = input.cpu().numpy()
        softB = softmax_y[:, 1, :, :]

        # Soft Dice
        sizePredNumpy = softB.sum()
        sizePred = torch.FloatTensor(1)
        sizePred.fill_(sizePredNumpy.item())

        if target[:, 1, :, :].sum() > 0:
            if (sizePred.numpy()[0] > upper_B.numpy()[0]):
                loss = ((sizePred - upper_B)**2)/(softB.shape[1]*softB.shape[2])
            elif (sizePred.numpy()[0] < lower_B.numpy()[0]):
                loss = ((sizePred - lower_B)**2)/(softB.shape[1]*softB.shape[2])
            else:
                loss = torch.FloatTensor(1)
                loss.fill_(0)
        else:
            loss = ((sizePred)**2)/(softB.shape[1]*softB.shape[2])

        lossT = loss/100

        if (np.isnan(loss.numpy()[0])):
            pdb.set_trace()

        return lossT.cuda()

    def backward(self, grad_output):
        input, target, lower_B, upper_B = self.saved_variables

        m = input.shape[2]*input.shape[3]

        # Compute the hard size of the prediction
        softmax_y = input.cpu().data.numpy()
        softB = softmax_y[:, 1, :, :]

        # Soft Dice
        sizePred = softB.sum()
        sizePredNumpy = softB.sum()
        sizePred = torch.FloatTensor(1)
        sizePred.fill_(sizePredNumpy.item())

        # TO-DO. Currently, the loss is weighted by a hard-coded value (100). Add this as input parameter
        if target[:, 1, :, :].sum().cpu().data.numpy() > 0:
            if (sizePred.numpy()[0] > upper_B.data.numpy()):
                lossValue = 2 * (sizePred-upper_B.data)/(100*m)
            elif (sizePred.numpy()[0] < lower_B.data.numpy()):
                lossValue = 2 * (sizePred-lower_B.data)/(100*m)
            else:
                lossValue = torch.FloatTensor(1)
                lossValue.fill_(0.0)
        else:
            lossValue = 2 * (sizePred)/(100*m)

        grad_inputA = np.zeros((softmax_y.shape[0], 1, softmax_y.shape[2], softmax_y.shape[3]), dtype='float32')
        grad_inputB = np.zeros((softmax_y.shape[0], 1, softmax_y.shape[2], softmax_y.shape[3]), dtype='float32')

        grad_inputB.fill(lossValue.numpy()[0])

        grad_input = np.concatenate((grad_inputA, grad_inputB), 1)

        return torch.Tensor(grad_input).cuda(), None, None, None
