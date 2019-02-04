#!/usr/bin/env python3.6

import random
import argparse
import warnings
from pathlib import Path
from pprint import pprint
from functools import partial
from typing import Any, Callable, List, Tuple

import numpy as np
import nibabel as nib
from numpy import unique as uniq
from skimage.io import imsave
from skimage.transform import resize
# from PIL import Image

from utils import mmap_, uc_, flatten_


def norm_arr(img: np.ndarray) -> np.ndarray:
    casted = img.astype(np.float32)
    shifted = casted - casted.min()
    norm = shifted / shifted.max()
    res = 255 * norm

    return res.astype(np.uint8)


def get_p_id(path: Path) -> str:
    """
    The patient ID
    """
    res = "_".join(path.stem.split('_')[:2])
    assert "Img" in res, res

    return res


def process_patient(img_p: Path, gt_p: Path,
                    dest_dir: Path, shape: Tuple[int, int], cr: int,
                    img_dir: str = "img", gt_dir: str = "gt") -> np.ndarray:
    p_id: str = get_p_id(img_p)
    assert p_id == get_p_id(gt_p)

    # Load the data
    img_nib = nib.load(str(img_p))
    x, y, z = img_nib.dataobj.shape
    dx, dy, dz = img_nib.header.get_zooms()

    # Make sure data is consistent with the description in the lineage
    assert (x, y, z) == (39, 305, 305), (x, y, z)
    assert 1.9 <= dx <= 2, dx
    assert dy == dz, (dy, dz)
    assert 1 <= dy <= 1.25, dy

    img = np.asarray(img_nib.dataobj)
    gt = np.asarray(nib.load(str(gt_p)).dataobj)

    assert img.shape == gt.shape
    assert img.dtype in [np.int16], img.dtype
    assert gt.dtype in [np.uint8], gt.dtype

    # Normalize and check data content
    norm_img = norm_arr(img)  # We need to normalize the whole 3d img, not 2d slices
    assert 0 == norm_img.min() and norm_img.max() == 255, (norm_img.min(), norm_img.max())
    assert norm_img.dtype == np.uint8

    norm_gt = gt.astype(np.uint8)
    assert set(uniq(gt)) == set(uniq(norm_gt)) == set([0, 1])
    del img  # Keep gt for sanity checks

    crop_img = norm_img[:, cr:-cr, :]
    crop_gt = norm_gt[:, cr:-cr, :]
    assert norm_gt.sum() == crop_gt.sum()  # Make sure we did not discard any part of the object
    del norm_img, norm_gt

    # Pad to get square slices
    _, ny, _ = crop_img.shape
    offset_x: int = (ny - x) // 2

    pad_img = np.zeros((ny, ny, z), dtype=np.uint8)
    pad_img[offset_x:offset_x + x, ...] = crop_img

    pad_gt = np.zeros((ny, ny, z), dtype=np.uint8)
    pad_gt[offset_x:offset_x + x, ...] = crop_gt
    del crop_img, crop_gt

    resize_: Callable = partial(resize, output_shape=(*shape, z), mode="constant", preserve_range=True, anti_aliasing=False)
    # resize_: Callable = lambda x, *_, **_2: x[cr:-cr, cr:-cr, :]

    resized_img = resize_(pad_img).astype(np.uint8)
    resized_gt = resize_(pad_gt, order=0)
    assert set(uniq(resized_gt)).issubset(set(uniq(gt))), (resized_gt.dtype, uniq(resized_gt))
    resized_gt = resized_gt.astype(np.uint8)
    del pad_img, pad_gt

    save_dir_img: Path = Path(dest_dir, img_dir)
    save_dir_gt: Path = Path(dest_dir, gt_dir)
    save_slices([resized_img, resized_gt], [save_dir_img, save_dir_gt], p_id)

    sizes = np.einsum("xyz->z", resized_gt, dtype=np.int64)

    return sizes


def save_slices(slices: List[np.ndarray], directories: List[Path], p_id: str) -> None:
    img, gt = slices
    x, y, z = img.shape
    assert x == y  # Want square slides

    for j in range(z):
        img_s = img[:, :, j]
        gt_s = gt[:, :, j]
        assert img_s.shape == gt_s.shape
        assert gt_s.dtype == np.uint8

        assert img_s.dtype == gt_s.dtype == np.uint8, img_s.dtype
        assert 0 <= img_s.min() and img_s.max() <= 255  # The range might be smaller bc of 3d norm

        for save_dir, data in zip(directories, [img_s, gt_s]):
                filename = f"{p_id}_{j}.png"
                save_dir.mkdir(parents=True, exist_ok=True)

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    imsave(str(Path(save_dir, filename)), data)


def main(args: argparse.Namespace):
    src_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)

    # Assume the cleaning up is done before calling the script
    assert src_path.exists()
    assert not dest_path.exists()

    # Get all the file names, avoid the temporal ones
    nii_paths: List[Path] = [p for p in src_path.rglob('*.nii')]
    assert len(nii_paths) % 2 == 0, "Uneven number of .nii, one+ pair is broken"

    # We sort now, but also id matching is checked while iterating later on
    img_nii_paths: List[Path] = sorted(p for p in nii_paths if "_Labels" not in str(p))
    gt_nii_paths: List[Path] = sorted(p for p in nii_paths if "_Labels" in str(p))
    assert len(img_nii_paths) == len(gt_nii_paths)
    paths: List[Tuple[Path, Path]] = list(zip(img_nii_paths, gt_nii_paths))

    print(f"Found {len(img_nii_paths)} pairs in total")
    pprint(paths[:5])

    validation_paths: List[Tuple[Path, Path]] = random.sample(paths, args.retain)
    training_paths: List[Tuple[Path, Path]] = [p for p in paths if p not in validation_paths]
    assert set(validation_paths).isdisjoint(set(training_paths))
    assert len(paths) == (len(validation_paths) + len(training_paths))

    for mode, _paths in zip(["train", "val"], [training_paths, validation_paths]):
        img_paths, gt_paths = zip(*_paths)  # type: Tuple[Any, Any]

        dest_dir = Path(dest_path, mode)
        print(f"Slicing {len(img_paths)} pairs to {dest_dir}")
        assert len(img_paths) == len(gt_paths)

        pfun = partial(process_patient, dest_dir=dest_dir, shape=args.shape, cr=args.crop)
        sizess = mmap_(uc_(pfun), zip(img_paths, gt_paths))
        # for paths in tqdm(list(zip(img_paths, gt_paths)), ncols=50):
        #     uc_(pfun)(paths)

        all_sizes = np.array(flatten_(sizess))
        all_pos = all_sizes[all_sizes > 0]

        print(f"sizes: min={np.min(all_pos)}, 5th={np.percentile(all_pos, 5):0.02f}, median={np.median(all_pos):0.0f}, " +
              f"mean={np.mean(all_pos):0.02f}, 95th={np.percentile(all_pos, 95):0.02f}, max={np.max(all_pos)}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Slicing parameters')
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)
    parser.add_argument('--crop', type=int, required=True, help="Will crop only the y axis")
    parser.add_argument('--shape', type=int, nargs=2, required=True)
    parser.add_argument('--retain', type=int, required=True)

    parser.add_argument('--img_dir', type=str, default="IMG")
    parser.add_argument('--gt_dir', type=str, default="GT")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed)

    print(args)

    return args


if __name__ == "__main__":
    args = get_args()
    random.seed(args.seed)

    main(args)
