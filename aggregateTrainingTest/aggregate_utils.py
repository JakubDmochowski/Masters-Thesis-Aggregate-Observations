import torch
import numpy as np


def length_to_range(lengths: list[int]):
    lengths = [0] + np.cumsum(lengths).tolist()
    return [range(a, b) for a, b in zip(lengths[:-1], lengths[1:])]
