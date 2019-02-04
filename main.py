#!/usr/bin/env python3.6

import argparse
import warnings
from pathlib import Path
from functools import reduce
from operator import add, itemgetter
from shutil import copytree, rmtree
from typing import Any, Callable, Dict, List, Tuple

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from networks import weights_init
from dataloader import get_loaders
from utils import map_
from utils import dice_coef, dice_batch, save_images, tqdm_, haussdorf
from utils import probs2one_hot, probs2class
from utils import depth


def setup(args, n_class: int) -> Tuple[Any, Any, Any, List[List[Callable]], List[List[float]], Callable]:
    print("\n>>> Setting up")
    cpu: bool = args.cpu or not torch.cuda.is_available()
    device = torch.device("cpu") if cpu else torch.device("cuda")

    if args.weights:
        if cpu:
            net = torch.load(args.weights, map_location='cpu')
        else:
            net = torch.load(args.weights)
        print(f">> Restored weights from {args.weights} successfully.")
    else:
        net_class = getattr(__import__('networks'), args.network)
        net = net_class(args.modalities, n_class).to(device)
        net.apply(weights_init)
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.l_rate, betas=(0.9, 0.99), amsgrad=False)

    # print(args.losses)
    list_losses = eval(args.losses)
    if depth(list_losses) == 1:  # For compatibility reasons, avoid changing all the previous configuration files
        list_losses = [list_losses]

    loss_fns: List[List[Callable]] = []
    for i, losses in enumerate(list_losses):
        print(f">> {i}th list of losses: {losses}")
        tmp: List[Callable] = []
        for loss_name, loss_params, _, _, fn, _ in losses:
            loss_class = getattr(__import__('losses'), loss_name)
            tmp.append(loss_class(**loss_params, fn=fn))
        loss_fns.append(tmp)

    loss_weights: List[List[float]] = [map_(itemgetter(5), losses) for losses in list_losses]

    scheduler = getattr(__import__('scheduler'), args.scheduler)(**eval(args.scheduler_params))

    return net, optimizer, device, loss_fns, loss_weights, scheduler


def do_epoch(mode: str, net: Any, device: Any, loaders: List[DataLoader], epc: int,
             list_loss_fns: List[List[Callable]], list_loss_weights: List[List[float]], C: int,
             savedir: str = "", optimizer: Any = None,
             metric_axis: List[int] = [1], compute_haussdorf: bool = False) \
        -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    assert mode in ["train", "val"]

    if mode == "train":
        net.train()
        desc = f">> Training   ({epc})"
    elif mode == "val":
        net.eval()
        desc = f">> Validation ({epc})"

    total_iteration: int = sum(len(loader) for loader in loaders)
    total_images: int = sum(len(loader.dataset) for loader in loaders)
    all_dices: Tensor = torch.zeros((total_images, C), dtype=torch.float32, device=device)
    batch_dices: Tensor = torch.zeros((total_iteration, C), dtype=torch.float32, device=device)
    loss_log: Tensor = torch.zeros((total_iteration), dtype=torch.float32, device=device)
    haussdorf_log: Tensor = torch.zeros((total_images, C), dtype=torch.float32, device=device)

    done_img: int = 0
    done_batch: int = 0
    tq_iter = tqdm_(total=total_iteration, desc=desc)
    for i, (loader, loss_fns, loss_weights) in enumerate(zip(loaders, list_loss_fns, list_loss_weights)):
        L: int = len(loss_fns)

        for data in loader:
            data[1:] = [e.to(device) for e in data[1:]]  # Move all tensors to device
            filenames, image, target = data[:3]
            labels = data[3:3 + L]
            bounds = data[3 + L:]
            assert len(labels) == len(bounds)

            B = len(image)

            # Reset gradients
            if optimizer:
                optimizer.zero_grad()

            # Forward
            pred_logits: Tensor = net(image)
            pred_probs: Tensor = F.softmax(pred_logits, dim=1)
            predicted_mask: Tensor = probs2one_hot(pred_probs.detach())  # Used only for dice computation

            assert len(bounds) == len(loss_fns) == len(loss_weights)
            ziped = zip(loss_fns, labels, loss_weights, bounds)
            losses = [w * loss_fn(pred_probs, label, bound) for loss_fn, label, w, bound in ziped]
            loss = reduce(add, losses)
            assert loss.shape == (), loss.shape

            # Backward
            if optimizer:
                loss.backward()
                optimizer.step()

            # Compute and log metrics
            loss_log[done_batch] = loss.detach()

            sm_slice = slice(done_img, done_img + B)  # Values only for current batch

            dices: Tensor = dice_coef(predicted_mask, target.detach())
            assert dices.shape == (B, C), (dices.shape, B, C)
            all_dices[sm_slice, ...] = dices

            if B > 1 and mode == "val":
                batch_dice: Tensor = dice_batch(predicted_mask, target.detach())
                assert batch_dice.shape == (C,), (batch_dice.shape, B, C)
                batch_dices[done_batch] = batch_dice

            if compute_haussdorf:
                haussdorf_res: Tensor = haussdorf(predicted_mask.detach(), target.detach())
                assert haussdorf_res.shape == (B, C)
                haussdorf_log[sm_slice] = haussdorf_res

            # Save images
            if savedir:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    predicted_class: Tensor = probs2class(pred_probs)
                    save_images(predicted_class, filenames, savedir, mode, epc)

            # Logging
            big_slice = slice(0, done_img + B)  # Value for current and previous batches

            dsc_dict = {f"DSC{n}": all_dices[big_slice, n].mean() for n in metric_axis}
            hauss_dict = {f"HD{n}": haussdorf_log[big_slice, n].mean() for n in metric_axis} if compute_haussdorf else {}
            batch_dict = {f"bDSC{n}": batch_dices[:done_batch, n].mean() for n in metric_axis} if B > 1 and mode == "val" else {}

            mean_dict = {"DSC": all_dices[big_slice, metric_axis].mean(),
                         "HD": haussdorf_log[big_slice, metric_axis].mean()} if len(metric_axis) > 1 else {}

            stat_dict = {**dsc_dict, **hauss_dict, **mean_dict, **batch_dict,
                         "loss": loss_log[:done_batch].mean()}
            nice_dict = {k: f"{v:.3f}" for (k, v) in stat_dict.items()}

            done_img += B
            done_batch += 1
            tq_iter.set_postfix({**nice_dict, "loader": str(i)})
            tq_iter.update(1)
    tq_iter.close()
    print(f"{desc} " + ', '.join(f"{k}={v}" for (k, v) in nice_dict.items()))

    return loss_log, all_dices, batch_dices, haussdorf_log


def run(args: argparse.Namespace) -> Dict[str, Tensor]:
    n_class: int = args.n_class
    lr: float = args.l_rate
    savedir: str = args.workdir
    n_epoch: int = args.n_epoch
    val_f: int = args.val_loader_id

    loss_fns: List[List[Callable]]
    loss_weights: List[List[float]]
    net, optimizer, device, loss_fns, loss_weights, scheduler = setup(args, n_class)
    train_loaders: List[DataLoader]
    val_loaders: List[DataLoader]
    train_loaders, val_loaders = get_loaders(args, args.dataset,
                                             args.batch_size, n_class,
                                             args.debug, args.in_memory)

    n_tra: int = sum(len(tr_lo.dataset) for tr_lo in train_loaders)  # Number of images in dataset
    l_tra: int = sum(len(tr_lo) for tr_lo in train_loaders)  # Number of iteration per epc: different if batch_size > 1
    n_val: int = sum(len(vl_lo.dataset) for vl_lo in val_loaders)
    l_val: int = sum(len(vl_lo) for vl_lo in val_loaders)

    best_dice: Tensor = torch.zeros(1).to(device).type(torch.float32)
    best_epoch: int = 0
    metrics = {"val_dice": torch.zeros((n_epoch, n_val, n_class), device=device).type(torch.float32),
               "val_batch_dice": torch.zeros((n_epoch, l_val, n_class), device=device).type(torch.float32),
               "val_loss": torch.zeros((n_epoch, l_val), device=device).type(torch.float32),
               "tra_dice": torch.zeros((n_epoch, n_tra, n_class), device=device).type(torch.float32),
               "tra_batch_dice": torch.zeros((n_epoch, l_tra, n_class), device=device).type(torch.float32),
               "tra_loss": torch.zeros((n_epoch, l_tra), device=device).type(torch.float32)}
    if args.compute_haussdorf:
        metrics["val_haussdorf"] = torch.zeros((n_epoch, n_val, n_class), device=device).type(torch.float32)

    print("\n>>> Starting the training")
    for i in range(n_epoch):
        # Do training and validation loops
        tra_loss, tra_dice, tra_batch_dice, _ = do_epoch("train", net, device, train_loaders, i,
                                                         loss_fns, loss_weights, n_class,
                                                         optimizer=optimizer,
                                                         metric_axis=args.metric_axis)
        with torch.no_grad():
            val_loss, val_dice, val_batch_dice, val_haussdorf = do_epoch("val", net, device, val_loaders, i,
                                                                         [loss_fns[val_f]], [loss_weights[val_f]],
                                                                         n_class,
                                                                         savedir=savedir,
                                                                         metric_axis=args.metric_axis,
                                                                         compute_haussdorf=args.compute_haussdorf)

        # Sort and save the metrics
        for k in metrics:
            assert metrics[k][i].shape == eval(k).shape, (metrics[k][i].shape, eval(k).shape)
            metrics[k][i] = eval(k)

        for k, e in metrics.items():
            np.save(Path(savedir, f"{k}.npy"), e.cpu().numpy())

        df = pd.DataFrame({"tra_loss": metrics["tra_loss"].mean(dim=1).cpu().numpy(),
                           "val_loss": metrics["val_loss"].mean(dim=1).cpu().numpy(),
                           "tra_dice": metrics["tra_dice"][:, :, -1].mean(dim=1).cpu().numpy(),
                           "val_dice": metrics["val_dice"][:, :, -1].mean(dim=1).cpu().numpy(),
                           "tra_batch_dice": metrics["tra_batch_dice"][:, :, -1].mean(dim=1).cpu().numpy(),
                           "val_batch_dice": metrics["val_batch_dice"][:, :, -1].mean(dim=1).cpu().numpy()})
        df.to_csv(Path(savedir, args.csv), float_format="%.4f", index_label="epoch")

        # Save model if better
        current_dice: Tensor = val_dice[:, args.metric_axis].mean()
        if current_dice > best_dice:
            best_epoch = i
            best_dice = current_dice
            if args.compute_haussdorf:
                best_haussdorf = val_haussdorf[:, args.metric_axis].mean()

            with open(Path(savedir, "best_epoch.txt"), 'w') as f:
                f.write(str(i))
            best_folder = Path(savedir, "best_epoch")
            if best_folder.exists():
                rmtree(best_folder)
            copytree(Path(savedir, f"iter{i:03d}"), Path(best_folder))
            torch.save(net, Path(savedir, "best.pkl"))

        optimizer, loss_fns, loss_weights = scheduler(i, optimizer, loss_fns, loss_weights)

        # if args.schedule and (i > (best_epoch + 20)):
        if args.schedule and (i % (best_epoch + 20) == 0):  # Yeah, ugly but will clean that later
            for param_group in optimizer.param_groups:
                lr *= 0.5
                param_group['lr'] = lr
                print(f'>> New learning Rate: {lr}')

        if i > 0 and not (i % 5):
            maybe_hauss = f', Haussdorf: {best_haussdorf:.3f}' if args.compute_haussdorf else ''
            print(f">> Best results at epoch {best_epoch}: DSC: {best_dice:.3f}{maybe_hauss}")

    # Because displaying the results at the end is actually convenient
    maybe_hauss = f', Haussdorf: {best_haussdorf:.3f}' if args.compute_haussdorf else ''
    print(f">> Best results at epoch {best_epoch}: DSC: {best_dice:.3f}{maybe_hauss}")
    for metric in metrics:
        if "val" in metric or "loss" in metric:  # Do not care about training values, nor the loss (keep it simple)
            print(f"\t{metric}: {metrics[metric][best_epoch].mean(dim=0)}")

    return metrics


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--dataset', type=str, required=True)
    # parser.add_argument('--weak_subfolder', type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--losses", type=str, required=True,
                        help="List of list of (loss_name, loss_params, bounds_name, bounds_params, fn, weight)")
    parser.add_argument("--folders", type=str, required=True,
                        help="List of list of (subfolder, transform, is_hot)")
    parser.add_argument("--network", type=str, required=True, help="The network to use")
    parser.add_argument("--grp_regex", type=str, required=True)
    parser.add_argument("--n_class", type=int, required=True)
    parser.add_argument("--metric_axis", type=int, nargs='+', required=True, help="Classes to display metrics")

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cpu", action='store_true')
    parser.add_argument("--in_memory", action='store_true')
    parser.add_argument("--schedule", action='store_true')
    parser.add_argument("--compute_haussdorf", action='store_true')
    parser.add_argument("--group", action='store_true', help="Group the patient slices together for validation. \
        Useful to compute the 3d dice, but might destroy the memory for datasets with a lot of slices per patient.")
    parser.add_argument("--group_train", action='store_true', help="Group the patient slices together for training. \
        Useful to compute the 3d dice, but might destroy the memory for datasets with a lot of slices per patient.")

    parser.add_argument('--n_epoch', nargs='?', type=int, default=200,
                        help='# of the epochs')
    parser.add_argument('--l_rate', nargs='?', type=float, default=5e-4,
                        help='Learning Rate')
    parser.add_argument("--scheduler", type=str, default="DummyScheduler")
    parser.add_argument("--scheduler_params", type=str, default="{}")
    parser.add_argument("--modalities", type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--weights", type=str, default='', help="Stored weights to restore")
    parser.add_argument("--training_folders", type=str, nargs="+", default=["train"])
    parser.add_argument("--val_loader_id", type=int, default=-1, help="""
                        Kinda housefiry at the moment. When we have several train loader (for hybrid training
                        for instance), wants only one validation loader. The way the dataloading creation is
                        written at the moment, it will create several validation loader on the same topfolder (val),
                        but with different folders/bounds ; which will basically duplicate the evaluation.
                        """)

    args = parser.parse_args()
    print("\n", args)

    return args


if __name__ == '__main__':
    run(get_args())
