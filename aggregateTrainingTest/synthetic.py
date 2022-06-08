import random
import torch
import numpy as np
from dataset import Observation
from typing import Callable
from itertools import chain
from dataset import Dataset
from aggregate_utils import length_to_range


def addNoise(data: torch.tensor) -> torch.tensor:
    return torch.tensor(data.numpy()).float()


def getExpectedValues(expected_y: torch.tensor, dataset: Dataset):
    data_x_indices = list(
        chain(*[obs.entries_indices for obs in dataset.observations]))
    return expected_y[data_x_indices]


def generateValues(data_x: torch.tensor, value_func: Callable) -> np.ndarray:
    return torch.tensor(np.array([value_func(x) for x in data_x.numpy()])).float()


def generateData(entry_no: int, dim_no: int, options: dict = {}) -> list:
    # returned data_x is a tensor shaped (entries, features)
    x_min = -10
    x_max = 10
    if ('x_min' in options):
        x_min = options["x_min"]
    if ('x_max' in options):
        x_max = options["x_max"]
    data_x = torch.tensor(
        np.random.uniform(x_min, x_max, entry_no * dim_no
                          ).reshape((entry_no, dim_no))).float()
    return data_x


def generateObservations(data_x: torch.tensor, num_observations: int, value_func: Callable, add_noise: bool) -> np.ndarray:
    # returned data_y is a tensor shaped (entries, values)
    entry_no = len(data_x)
    meta = np.linspace(0, entry_no, entry_no, endpoint=False, dtype=int)
    random.shuffle(meta)
    meta = np.array_split(meta, num_observations)
    meta = [Observation(x, i) for i, x in enumerate(meta)]
    data_y = generateValues(data_x, value_func)
    if(add_noise is True):
        data_y = addNoise(data_y)
    data_y = data_y
    obs_y = torch.tensor([torch.index_select(
        data_x, 0, torch.tensor(obs.entries_indices)).mean(axis=0) for obs in meta]).float()
    return [data_y, obs_y, meta]
