#!/usr/env/bin python3.6

import pdb
from typing import List, Tuple

import torch
import numpy as np
from torch import Tensor, einsum

from utils import simplex, sset, probs2one_hot, one_hot


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = target[:, self.idc, ...].type(torch.float32)

        loss = - einsum("bcwh,bcwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class NegCrossEntropy(CrossEntropy):
    """
    Apply the cross-entropy ONLY if the whole image is the selected class.
    This is useful to supervise the background class when we have weak labels.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, bounds: Tensor) -> Tensor:
        _, _, w, h = probs.shape
        trimmed: Tensor = target[:, self.idc, ...]
        full_img: Tensor = einsum("bcwh->b", trimmed) == (w * h)  # List of images that are fully covered

        if full_img.any():
            where = torch.nonzero(full_img).flatten()
            return super().__call__(probs[where], target[where], bounds[where])

        return torch.zeros(1).to(probs.device)


class NaivePenalty():
    def __init__(self, **kwargs):
        self.idc: List[int] = kwargs["idc"]
        self.C = len(self.idc)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])

    def __call__(self, probs: Tensor, target: Tensor, bounds: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)
        assert probs.shape == target.shape

        b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        k = bounds.shape[2]  # scalar or vector
        value: Tensor = self.__fn__(probs[:, self.idc, ...])
        lower_b = bounds[:, self.idc, :, 0]
        upper_b = bounds[:, self.idc, :, 1]

        assert value.shape == (b, self.C, k), value.shape
        assert lower_b.shape == upper_b.shape == (b, self.C, k), lower_b.shape

        too_big: Tensor = (value > upper_b).type(torch.float32)
        too_small: Tensor = (value < lower_b).type(torch.float32)

        big_pen: Tensor = (value - upper_b) ** 2
        small_pen: Tensor = (value - lower_b) ** 2

        res = too_big * big_pen + too_small * small_pen

        loss: Tensor = res / (w * h)

        return loss.mean()


class BatchNaivePenalty():
    """
    Used to supervise the size of the batch (3d patient). Will sum all the exact bounds, and add the margins itself
    """
    def __init__(self, **kwargs):
        self.idc: List[int] = kwargs["idc"]
        self.C = len(self.idc)
        self.margin: float = kwargs["margin"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])

    def __call__(self, probs: Tensor, target: Tensor, bounds: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)
        assert probs.shape == target.shape

        target_sizes: Tensor = bounds[:, self.idc, :, 1]  # Dim of 1, upper and lower are the same
        volume_size: Tensor = einsum("bck->ck", target_sizes)

        lower_b = volume_size * (1 - self.margin)
        upper_b = volume_size * (1 + self.margin)

        _, _2, w, h = probs.shape  # type: Tuple[int, int, int, int]
        k = bounds.shape[2]  # scalar or vector
        value: Tensor = self.__fn__(probs[:, self.idc, ...]).sum(dim=0)
        assert value.shape == (self.C, k), value.shape
        assert lower_b.shape == upper_b.shape == (self.C, k), lower_b.shape

        too_big: Tensor = (value > upper_b).type(torch.float32)
        too_small: Tensor = (value < lower_b).type(torch.float32)

        big_pen: Tensor = (value - upper_b) ** 2
        small_pen: Tensor = (value - lower_b) ** 2

        res = too_big * big_pen + too_small * small_pen

        loss: Tensor = res / (w * h)

        return loss.mean()


class Pathak(CrossEntropy):
    def __init__(self, **kwargs):
        self.ignore = kwargs["ignore"]
        super().__init__(**kwargs)

    def __call__(self, probs: Tensor, target: Tensor, bounds: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target) and sset(target, [0, 1])
        assert probs.shape == target.shape

        with torch.no_grad():
            fake_mask: Tensor = torch.zeros_like(probs)
            for i in range(len(probs)):
                fake_mask[i] = self.pathak_generator(probs[i], target[i], bounds[i])
                self.holder_size = fake_mask[i].sum()

        return super().__call__(probs, fake_mask, bounds)

    def pathak_generator(self, probs: Tensor, target: Tensor, bounds) -> Tensor:
        _, w, h = probs.shape

        # Replace the probabilities with certainty for the few weak labels that we have
        weak_labels = target[...]
        weak_labels[self.ignore, ...] = 0
        assert not simplex(weak_labels) and simplex(target)
        lower, upper = bounds[-1]

        labeled_pixels = weak_labels.any(axis=0)
        assert w * h == (labeled_pixels.sum() + (~labeled_pixels).sum())  # make sure all pixels are covered
        scribbled_probs = weak_labels + einsum("cwh,wh->cwh", probs, ~labeled_pixels)
        assert simplex(scribbled_probs)

        u: Tensor
        max_iter: int = 100
        lr: float = 0.00005
        b: Tensor = Tensor([-lower, upper])
        beta: Tensor = torch.zeros(2, torch.float32)
        f: Tensor = torch.zeros(2, *probs.shape)
        f[0, ...] = -1
        f[1, ...] = 1

        for i in range(max_iter):
            exped = - einsum("i,icwh->cwh", beta, f).exp()
            u_star = einsum('cwh,cwh->cwh', probs, exped)
            u_star /= u_star.sum(axis=0)
            assert simplex(u_star)

            d_beta = einsum("cwh,icwh->i", u_star, f) - b
            n_beta = torch.max(torch.zeros_like(beta), beta + lr * d_beta)

            u = u_star
            beta = n_beta

        return probs2one_hot(u)


class WeightedCrossEntropy():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = target[:, self.idc, ...].type(torch.float32)

        w: Tensor = 1 / ((einsum("bcwh->bc", mask).type(torch.float32) + 1e-10) ** 2)
        loss = - einsum("bc,bcwh,bcwh->", w, mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class GeneralizedDice():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        w: Tensor = 1 / ((einsum("bcwh->bc", tc).type(torch.float32) + 1e-10) ** 2)
        intersection: Tensor = w * einsum("bcwh,bcwh->bc", pc, tc)
        union: Tensor = w * (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))

        divided: Tensor = 1 - 2 * (einsum("bc->b", intersection) + 1e-10) / (einsum("bc->b", union) + 1e-10)

        loss = divided.mean()

        return loss


class DiceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        intersection: Tensor = einsum("bcwh,bcwh->bc", pc, tc)
        union: Tensor = (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))

        divided: Tensor = 1 - (2 * intersection + 1e-10) / (union + 1e-10)

        loss = divided.mean()

        return loss


class SurfaceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_maps: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum("bcwh,bcwh->bcwh", pc, dc)

        loss = multipled.mean()

        return loss


class PathakLoss(CrossEntropy):
    def __init__(self, **kwargs):
        self.mask_idc: List[int] = kwargs["mask_idc"]
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, bounds: Tensor) -> Tensor:
        assert simplex(probs)
        assert probs.shape == target.shape
        assert len(self.mask_idc) == 1, "Cannot handle more at the time, I guess"

        b, c, w, h = probs.shape

        fake_probs: Tensor = torch.zeros_like(probs, dtype=torch.float32)
        for i in range(len(probs)):
            low: Tensor = bounds[i, self.mask_idc][0, 0, 0]
            high: Tensor = bounds[i, self.mask_idc][0, 0, 1]

            res = self.pathak_generator(probs[i].detach(), target[i].detach(), low, high)
            assert simplex(res, axis=0)
            assert res.shape == (c, w, h)

            fake_probs[i] = res
        fake_mask: Tensor = probs2one_hot(fake_probs)
        assert fake_mask.shape == probs.shape == target.shape

        return super().__call__(probs, fake_mask, bounds)

    def pathak_generator(self, probs: Tensor, weak_labels: Tensor, a: Tensor, b: Tensor) -> Tensor:
        with torch.no_grad():
            C, W, H = probs.shape
            assert C == 2  # Keep it simple for now
            assert probs.shape == weak_labels.shape
            assert not probs.requires_grad and not weak_labels.requires_grad

            if (a == b == torch.zeros_like(a)).all():
                # Fails miserably to supress when 0 < size < 0
                a = torch.ones_like(a) * -1

            assert simplex(probs, axis=0), probs.sum(dim=0)
            assert simplex(weak_labels, axis=0), weak_labels.sum(dim=0)

            trimmed_labels = torch.zeros_like(weak_labels)
            trimmed_labels[self.mask_idc, ...] = weak_labels[self.mask_idc, ...]
            trimmed_labels = trimmed_labels.type(torch.uint8)  # Required to do any()
            assert not simplex(trimmed_labels, axis=0)  # If it is simplex, means we are using all classes
            # Which is either a mistake (use also dummy background class) or equivalent to FS ; to this is pointless

            unlabeled_pixels = ~(trimmed_labels.any(dim=0))
            assert (W * H) == (unlabeled_pixels.sum() + (~unlabeled_pixels).sum())
            assert (~unlabeled_pixels).sum() == trimmed_labels.sum() < (W * H)

            # Replace the probabilities with certainty for the few weak labels that we have
            fixed_probs = trimmed_labels.type(torch.float32) + probs * unlabeled_pixels.type(torch.float32)
            assert simplex(fixed_probs, axis=0)

            u: Tensor
            max_iter: int = 500
            beta1: Tensor = torch.zeros(1, device=fixed_probs.device, dtype=torch.float32)
            beta2: Tensor = torch.zeros(1, device=fixed_probs.device, dtype=torch.float32)
            _zero: Tensor = torch.zeros_like(beta1)
            lr: float = 0.00005
            f: Tensor = torch.zeros_like(fixed_probs)
            f[self.mask_idc, ...] = 1

            for i in range(max_iter):
                u_star: Tensor = fixed_probs * torch.exp(-beta1 * f + beta2 * f)
                u_star /= u_star.sum(dim=0)
                assert u_star.dtype == torch.float32
                try:
                    assert simplex(u_star, axis=0)
                except AssertionError:
                    pdb.set_trace()

                summed: Tensor = einsum("cwh,cwh->", u_star, f)
                d_beta1 = (summed - b).item()
                d_beta2 = (- summed + a).item()

                n_beta1 = torch.max(_zero, beta1 + lr * d_beta1)
                n_beta2 = torch.max(_zero, beta2 + lr * d_beta2)

                u = u_star
                if (torch.abs(beta1 - n_beta1) / (beta1 + 1.e-10)) < 0.001:
                    break

                beta1 = n_beta1
                beta2 = n_beta2

            assert simplex(u, axis=0)
            assert u.shape == (C, W, H)

            show = False
            if show and b.sum() == 0:
                import matplotlib.pyplot as plt

                print(f"Took {i} iterations to compute u, a={a}, b={b}")
                figs = [(weak_labels[1], "Weak labels"),
                        (trimmed_labels[1], "Trimmed labels"),
                        (probs[1], "Init probs"),
                        (fixed_probs[1], "Fixed probs"),
                        (u[1], "Fake probs")]

                _, axes = plt.subplots(nrows=1, ncols=len(figs))

                for axe, fig in zip(axes, figs):
                    axe.set_title(fig[1])
                    axe.imshow(fig[0].cpu().numpy())
                plt.show()

            return u


class PathakUpper(CrossEntropy):
    def __init__(self, **kwargs):
        self.mask_idc: List[int] = kwargs["mask_idc"]
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, bounds: Tensor) -> Tensor:
        assert simplex(probs)
        assert probs.shape == target.shape
        assert len(self.mask_idc) == 1, "Cannot handle more at the time, I guess"

        b, c, w, h = probs.shape

        fake_probs: Tensor = torch.zeros_like(probs, dtype=torch.float32)
        for i in range(len(probs)):
            low: Tensor = bounds[i, self.mask_idc][0, 0, 0]
            high: Tensor = bounds[i, self.mask_idc][0, 0, 1]

            res = self.pathak_generator(probs[i].detach(), target[i].detach(), low, high)
            assert simplex(res, axis=0)
            assert res.shape == (c, w, h)

            fake_probs[i] = res
        fake_mask: Tensor = probs2one_hot(fake_probs)
        assert fake_mask.shape == probs.shape == target.shape

        return super().__call__(probs, fake_mask, bounds)

    def pathak_generator(self, probs: Tensor, weak_labels: Tensor, _: Tensor, b: Tensor) -> Tensor:
        with torch.no_grad():
            C, W, H = probs.shape
            assert C == 2  # Keep it simple for now
            assert probs.shape == weak_labels.shape
            assert not probs.requires_grad and not weak_labels.requires_grad

            assert simplex(probs, axis=0), probs.sum(dim=0)
            assert simplex(weak_labels, axis=0), weak_labels.sum(dim=0)

            trimmed_labels = torch.zeros_like(weak_labels)
            trimmed_labels[self.mask_idc, ...] = weak_labels[self.mask_idc, ...]
            trimmed_labels = trimmed_labels.type(torch.uint8)  # Required to do any()
            assert not simplex(trimmed_labels, axis=0)  # If it is simplex, means we are using all classes
            # Which is either a mistake (use also dummy background class) or equivalent to FS ; to this is pointless

            unlabeled_pixels = ~(trimmed_labels.any(dim=0))
            assert (W * H) == (unlabeled_pixels.sum() + (~unlabeled_pixels).sum())
            assert (~unlabeled_pixels).sum() == trimmed_labels.sum() < (W * H)

            # Replace the probabilities with certainty for the few weak labels that we have
            fixed_probs = trimmed_labels.type(torch.float32) + probs * unlabeled_pixels.type(torch.float32)
            assert simplex(fixed_probs, axis=0)

            u: Tensor
            max_iter: int = 500
            beta1: Tensor = torch.zeros(1, device=fixed_probs.device, dtype=torch.float32)
            _zero: Tensor = torch.zeros_like(beta1)
            lr: float = 0.00005
            f: Tensor = torch.zeros_like(fixed_probs)
            f[self.mask_idc, ...] = 1

            for i in range(max_iter):
                u_star: Tensor = fixed_probs * torch.exp(-beta1 * f)
                u_star /= u_star.sum(dim=0)
                assert u_star.dtype == torch.float32
                try:
                    assert simplex(u_star, axis=0)
                except AssertionError:
                    pdb.set_trace()

                summed: Tensor = einsum("cwh,cwh->", u_star, f)
                d_beta1 = (summed - b).item()

                n_beta1 = torch.max(_zero, beta1 + lr * d_beta1)

                u = u_star
                if (torch.abs(beta1 - n_beta1) / (beta1 + 1.e-10)) < 0.001:
                    break

                beta1 = n_beta1

            assert simplex(u, axis=0)
            assert u.shape == (C, W, H)

            show = False
            if show and b.sum() == 0:
                import matplotlib.pyplot as plt

                print(f"Took {i} iterations to compute u, b={b}")
                figs = [(weak_labels[1], "Weak labels"),
                        (trimmed_labels[1], "Trimmed labels"),
                        (probs[1], "Init probs"),
                        (fixed_probs[1], "Fixed probs"),
                        (u[1], "Fake probs")]

                _, axes = plt.subplots(nrows=1, ncols=len(figs))

                for axe, fig in zip(axes, figs):
                    axe.set_title(fig[1])
                    axe.imshow(fig[0].cpu().numpy())
                plt.show()

            return u
