import torch
import numpy as np
from data.dataset import Observation
from typing import Callable
from data.data_utils import generateValues


def addNoise(data: torch.tensor) -> torch.tensor:
    return torch.tensor(data.numpy()).float()


def generatePoints(entry_no: int, dim_no: int, options: dict) -> torch.tensor:
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


def generateObservations(data_y: torch.tensor, num_observations: int, value_func: Callable, add_noise: bool) -> list[torch.tensor, list[Observation]]:
    # returned data_y is a tensor shaped (entries, values)
    entry_no = len(data_y)
    meta = np.linspace(0, entry_no, entry_no, endpoint=False, dtype=int)
    np.random.shuffle(meta)
    meta = np.array_split(meta, num_observations)
    meta = [Observation(x, i) for i, x in enumerate(meta)]
    if(add_noise is True):
        data_y = addNoise(data_y)
    obs_y = torch.stack([torch.index_select(
        data_y, 0, torch.tensor(obs.entries_indices)).mean(axis=0) for obs in meta]).float()
    return [obs_y, meta]


def generateData(entry_no: int, num_observations: int, dim_no: int, value_func: Callable, add_noise: bool, options: dict = {}) -> list[torch.tensor, torch.tensor, torch.tensor, list[Observation]]:
    data_x = generatePoints(entry_no, dim_no, options)
    data_y = generateValues(data_x, value_func)
    obs_y, meta = generateObservations(
        data_y, num_observations, value_func, add_noise)
    return [data_x, data_y, obs_y, meta]
