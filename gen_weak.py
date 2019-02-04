#!/usr/bin/env python3

import random
import argparse
from typing import Callable, List, Tuple
from pathlib import Path
from pprint import pprint
from argparse import Namespace
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import ndimage
from PIL import Image, ImageDraw

from utils import mmap_, map_


def centroid_strat(orig_mask: np.ndarray, filename: str, filling: int) -> Tuple[np.ndarray, int]:
    assert set(np.unique(orig_mask)).issubset({False, True})

    size: int = orig_mask.sum()
    if size:  # Positive images
        res_img: Image.Image = Image.new("L", orig_mask.shape, 0)
        canvas = ImageDraw.Draw(res_img)

        centroid: Tuple[float, float] = ndimage.measurements.center_of_mass(orig_mask)

        # Of course the coordinates are inverted
        cx, cy = int(centroid[0]), int(centroid[1])
        cx, cy = cy, cx
        if args.verbose:
            print(centroid, cx, cy)

        width: int = args.width
        dw: int = int(width / 2)
        rx: int
        ry: int
        if args.r > 0:
            rx, ry = random.randrange(-args.r, args.r), random.randrange(-args.r, args.r)
        else:
            rx, ry = 0, 0
        canvas.ellipse([cx - dw + rx, cy - dw + ry, cx + dw + rx, cy + dw + ry], fill=filling)

        # Remove overflow if needed
        masked_res: Image.Image = Image.fromarray(np.array(res_img) * orig_mask, mode='L')

        # Sanity check: we do not want the label to go over the border
        result_mask = np.array(masked_res) == filling
        inter: np.ndarray = orig_mask & result_mask  # should be > 1

        if inter.sum() < 1:  # I assume this case appears only for images that are too small and r too big
            # So it makes sense to use the orignal segmentation as ground truth
            print(f"No overlap, using orignal mask {filename}")
            masked_res = Image.fromarray(orig_mask.astype(np.uint8), mode='L')
    else:
        masked_res = Image.fromarray(orig_mask.astype(np.uint8), mode='L')

    return masked_res, size


def erosion_strat(orig_mask: np.ndarray, filename: str, filling: int) -> Tuple[np.ndarray, int]:
    res_img: Image.Image = Image.new("L", orig_mask.shape, 0)

    size: int = orig_mask.sum()
    if size:  # Positive images
        struct2 = ndimage.generate_binary_structure(2, 3)
        # print(struct2.shape, orig_mask.shape)

        gt_eroded = orig_mask[...]
        iter = 10
        while True:  # do while du pauvre
            if iter == 0:
                gt_eroded = orig_mask[...]
                print(f"Using orignal structure for {filename} (size {orig_mask.sum()})")
                # plt.imshow(gt_eroded)
                # plt.show()
                break
            gt_eroded = ndimage.binary_erosion(orig_mask, structure=struct2, iterations=iter).astype(orig_mask.dtype)

            if gt_eroded.sum() > 0:
                break
            iter -= 1

        res = gt_eroded.astype(np.uint8)
        res[res == 1] = filling
        res_img = Image.fromarray(res, mode="L")
    return res_img, size


def random_strat(orig_mask: np.ndarray, filename: str, filling: int) -> Tuple[np.ndarray, int]:
    res_img: Image.Image = Image.new("L", orig_mask.shape, 0)

    size: int = orig_mask.sum()
    if size:  # Positive images
        canvas = ImageDraw.Draw(res_img)
        xs, ys = np.where(orig_mask == 1)
        # print(len(xs), len(ys))
        assert len(xs) == len(ys)
        random_index: int = np.random.randint(len(xs))
        rx, ry = xs[random_index], ys[random_index]
        # Of course the coordinates are inverted
        rx, ry = ry, rx
        # print(centroid, rx, ry)

        width: int = args.width
        dw: int = int(width / 2)
        canvas.ellipse([rx - dw, ry - dw, rx + dw, ry + dw], fill=filling)

        # Remove overflow if needed
        masked_res: Image.Image = Image.fromarray((np.array(res_img) * orig_mask).astype(np.uint8), mode='L')

        res_img = masked_res
    return res_img, size


def box_strat(orig_mask: np.ndarray, filename: str, filling: int) -> Tuple[np.ndarray, int]:
    orig_arr: np.ndarray = np.array(orig_mask, dtype=np.uint8)
    res_arr: np.ndarray = np.zeros_like(orig_arr)
    assert orig_arr.dtype == res_arr.dtype

    margin: int = args.margin

    size: int = orig_mask.sum()
    if size:  # Positive images
        coords = np.argwhere(orig_arr)

        x1, y1 = np.maximum(coords - margin, 0).min(axis=0)
        x2, y2 = np.minimum(coords + margin, orig_arr.shape).max(axis=0)

        res_arr[x1:x2 + 1, y1:y2 + 1] = filling

    res = Image.fromarray(res_arr, mode='L')

    return res, size


def weaken_img(pn: Tuple, strategy: Callable) -> Tuple[int, int]:
    # print(f"Processing {n}")
    p: str
    n: str
    p, n = pn

    img: Image.Image = Image.open(p)
    try:
        assert set(np.unique(img)).issubset({0, 1, 2, 3})
    except AssertionError:
        print(np.unique(img))
        raise
    if args.verbose:
        plt.imshow(img)
        plt.show()

    selected_class: int = args.selected_class
    filling: int = args.filling
    ni: np.ndarray = np.array(img) == selected_class  # Keep only background and LV, as booleans
    assert set(np.unique(ni)).issubset({False, True})

    # Do the magic
    res_img, size = strategy(ni, n, filling)

    # Final checks, we do not want the label to go over the border
    res_arr: np.ndarray = np.array(res_img)
    rb = np.array(res_arr) == filling
    inter: np.ndarray = ni & rb
    inter_neg: np.ndarray = (~ni) & rb
    try:
        assert res_arr.shape == ni.shape, (res_arr.shape, ni.shape)
        assert set(np.unique(rb)).issubset({False, True}), np.unique(rb)
        assert set(np.unique(res_arr)).issubset({0, filling}), np.unique(res_arr)
        assert rb.sum() <= ni.sum() or args.allow_bigger, (rb.sum(), ni.sum())
        assert inter_neg.sum() == 0 or args.allow_overflow, inter_neg.sum()  # No overflow over the border
        assert inter.sum() > 0 or size == 0, (inter.sum(), size == 0)  # At least some overlap
    except AssertionError:
        # print(res_arr.shape, ni.shape)
        # print(np.unique(rb), np.unique(res_arr))
        # print(rb.sum(), ni.sum())
        # print(inter_neg.sum())
        # print(inter.sum())
        _, axes = plt.subplots(nrows=1, ncols=2)

        for axe, fig in zip(axes, [np.array(img), res_arr]):
            axe.imshow(fig)
        plt.show()
        raise

    save_path = Path(args.base_folder, args.save_subfolder, n)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    res_img.save(save_path)

    return size, res_arr.sum()


def main(args: Namespace) -> None:
    inputs: List[Path] = list(Path(args.base_folder, args.GT_subfolder).glob(args.regex))
    names: List[str] = [p.name for p in inputs]
    print(f"Found {len(names)} images to weaken")
    if args.verbose:
        pprint(names[:10])

    strategy: Callable = eval(args.strategy)
    strat: Callable = partial(weaken_img, strategy=strategy)

    # sizes: np.ndarray = np.zeros(len(inputs), dtype=np.uint32)
    # for i, (pn) in tqdm(enumerate(zip(inputs, names)), ncols=100, total=len(names)):
    #     sizes[i] = strat(pn)
    orig_sizes, new_sizes = map_(np.asarray, zip(*mmap_(strat, zip(inputs, names))))
    assert len(orig_sizes) == len(new_sizes) == len(names)

    try:
        print("Orig sizes: (min, mean, max)", orig_sizes[orig_sizes > 0].min(), orig_sizes.mean(), orig_sizes.max())
        print(f"Annotated {new_sizes.sum()} pixels for {len(new_sizes)} images")
    except ValueError:
        pass


def get_args() -> Namespace:
    parser = argparse.ArgumentParser(description='Dataset params')
    parser.add_argument("--base_folder", type=str, required=True)
    parser.add_argument("--save_subfolder", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--strategy", type=str, required=True)
    parser.add_argument("--selected_class", type=int, required=True,
                        help="Default used to be 3")
    parser.add_argument("--filling", type=int, required=True,
                        help="Default used to be 3")

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--allow_bigger", action="store_true")
    parser.add_argument("--allow_overflow", action="store_true")

    parser.add_argument("--GT_subfolder", default='gt', type=str)
    parser.add_argument("--regex", type=str, default="*.png")
    parser.add_argument("--r", type=int, default=0)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--margin", type=int, default=0)

    args = parser.parse_args()
    print(args)

    return args


if __name__ == "__main__":
    args: Namespace = get_args()
    random.seed(args.seed)
    main(args)
