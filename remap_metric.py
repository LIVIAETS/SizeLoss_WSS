#!/usr/bin/env python3.7

from sys import argv
import numpy as np


def main() -> None:
    target: str = argv[1]
    dic = eval(argv[2])

    src_data: np.ndarray = np.load(target)
    shape = src_data.shape

    C: int = max(dic.values()) + 1
    dest_data: np.ndarray = np.zeros((*shape[:-1], C), dtype=src_data.dtype)
    # print(target, shape, dic, dest_data.shape)

    for k, v in dic.items():
        dest_data[..., v] = src_data[..., k]

    np.save(target, dest_data)


if __name__ == "__main__":
    main()
