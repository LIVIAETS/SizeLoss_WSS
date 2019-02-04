#!/usr/bin/env python3.6

import argparse
import warnings
from typing import List
from pathlib import Path

import torch
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader

from dataloader import SliceDataset
from utils import save_images, map_, tqdm_, probs2class, uniq


def runInference(args: argparse.Namespace):
    print('>>> Loading model')
    net = torch.load(args.model_weights)
    device = torch.device("cuda")
    net.to(device)

    print('>>> Loading the data')
    batch_size: int = args.batch_size
    num_classes: int = args.num_classes

    transform = transforms.Compose([
        lambda img: np.array(img)[np.newaxis, ...],
        lambda nd: nd / 255,  # max <= 1
        lambda nd: torch.tensor(nd, dtype=torch.float32)
    ])

    folders: List[Path] = [Path(args.data_folder)]
    names: List[str] = map_(lambda p: str(p.name), folders[0].glob("*.png"))
    dt_set = SliceDataset(names,
                          folders,
                          transforms=[transform],
                          debug=False,
                          C=num_classes)
    loader = DataLoader(dt_set,
                        batch_size=batch_size,
                        num_workers=batch_size + 2,
                        shuffle=False,
                        drop_last=False)

    print('>>> Starting the inference')
    savedir: str = args.save_folder
    total_iteration = len(loader)
    desc = f">> Inference"
    tq_iter = tqdm_(enumerate(loader), total=total_iteration, desc=desc)
    with torch.no_grad():
        for j, (filenames, image, _) in tq_iter:
            image = image.to(device)

            pred_logits: Tensor = net(image)
            pred_probs: Tensor = F.softmax(pred_logits, dim=1)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                predicted_class: Tensor = probs2class(pred_probs)
                save_images(predicted_class, filenames, savedir, "", 0)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Inference parameters')
    parser.add_argument('--data_folder', type=str, required=True, help="The folder containing the images to predict")
    parser.add_argument('--save_folder', type=str, required=True)
    parser.add_argument('--model_weights', type=str, required=True)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=10)
    args = parser.parse_args()

    print(args)

    return args


if __name__ == '__main__':
    args = get_args()
    runInference(args)
