# Constrained-CNN losses for weakly supervised segmentation
Code of our submission https://openreview.net/forum?id=BkIBHb2sG at MIDL 2018

To run it, simply run `main_MIDL.py` (python3.6+, the requirements are specified in the `requirements.txt` file).

The partial ground truth that we used are provided, but not the original dataset: https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html
You will need to download and pre-process it yourselves first.

The code was developed for PyTorch 0.3.1, and has been modified slightly to work with PyTorch 0.4. However, a lot of cleanup (removing the variables for instance) still need to be done.

## Loss functions
The loss functions are located in `losses.py`, and are defined as autograd functions. We implemented manually both the forward and the backward passes with numpy. We use a batch size of 1, and the code might need to be modified before working for more.

The inputs (predictions, labels and weak labels) are all represented as 4-D tensors:
```python
b, c, w, h = input.shape
assert target.shape == input.shape
assert weakLabels.shape == (b, 1, w, h)
```
`b` is the batch size, `c` the number of classes (2), and `w, h` the image size. Since this is a binary problem, the two classes are complementary (minus the rounding errors), both for the predictions and labels:
```python
assert np.allclose(input[:, 0, ...].cpu().numpy(), 1 - input[:, 1, ...].cpu().numpy(), atol=1e-2)
```