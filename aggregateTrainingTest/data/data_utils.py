import torch
from dataset import Dataset
from typing import Callable
import numpy as np
from itertools import chain


def observationSubsetFor(data: torch.tensor, dataset: Dataset):
    data_indices = list(
        chain(*[obs.entries_indices for obs in dataset.observations]))
    return data[data_indices]


def generateValues(data_x: torch.tensor, value_func: Callable) -> np.ndarray:
    # returned data_y is a tensor shaped (entries, values)
    return torch.tensor(np.array([value_func(x) for x in data_x.numpy()])).float()
