import torch
import numpy as np
from data.dataset import Observation
from typing import Callable
from data.data_utils import generate_values

X_MIN = -10
X_MAX = 10


def add_noise(data: torch.tensor) -> torch.tensor:
    return torch.tensor(data.numpy()).float()


def generate_points(entry_no: int, dim_no: int, options: dict) -> torch.tensor:
    # returned data_x is a tensor shaped (entries, features)
    X_MIN = -10
    X_MAX = 10
    if 'x_min' in options:
        X_MIN = options["x_min"]
    if 'x_max' in options:
        X_MAX = options["x_max"]
    data_x = torch.tensor(
        np.random.uniform(X_MIN, X_MAX, entry_no * dim_no
                          ).reshape((entry_no, dim_no))).float()
    return data_x


def aggregate_by(data_y: torch.tensor, meta: list[Observation]):
    def get_entries(indices):
        return torch.index_select(data_y, 0, torch.tensor(indices))

    def aggregate(entries: torch.tensor):
        return entries.mean(axis=0)

    obs_y = torch.stack([aggregate(get_entries(obs.entries_indices))
                        for obs in meta]).float()
    return obs_y


def generate_independent_observations(data_y: torch.tensor, num_observations: int, do_add_noise: bool) -> list[
        torch.tensor, list[Observation]]:
    # returned data_y is a tensor shaped (entries, values)
    entry_no = len(data_y)
    meta = np.linspace(0, entry_no, entry_no, endpoint=False, dtype=int)
    np.random.shuffle(meta)
    meta = np.array_split(meta, num_observations)
    meta = [Observation(x, i) for i, x in enumerate(meta)]
    if do_add_noise is True:
        data_y = add_noise(data_y)

    obs_y = aggregate_by(data_y, meta)
    return [obs_y, meta]


def generate_dependent_observations(data_x: torch.tensor, data_y: torch.tensor, num_observations: int,
                                    do_add_noise: bool) -> list[
        torch.tensor, list[Observation]]:
    # returned data_y is a tensor shaped (entries, values)
    meta = np.linspace(X_MIN, X_MAX, num_observations,
                       endpoint=False, dtype=float)
    meta = [torch.logical_and((data_x[:, 0] < curr), (data_x[:, 0] >= prev)).nonzero(as_tuple=True)[0] for prev, curr in
            zip(meta, meta[1:])]
    meta = [obs.numpy().tolist() for obs in meta if obs.size(dim=0)]
    meta = [Observation(x, i) for i, x in enumerate(meta)]
    if do_add_noise is True:
        data_y = add_noise(data_y)

    obs_y = aggregate_by(data_y, meta)

    return [obs_y, meta]


def generate_data(entry_no: int, num_observations: int, dim_no: int, value_func: Callable, do_add_noise: bool,
                  options: dict = {}) -> list[torch.tensor, torch.tensor, torch.tensor, list[Observation]]:
    data_x = generate_points(entry_no, dim_no, options)
    data_y = generate_values(data_x, value_func)
    obs_y, meta = generate_independent_observations(
        data_y, num_observations, do_add_noise)
    # obs_y, meta = generate_dependent_observations(data_x, data_y, num_observations, do_add_noise)
    # obs_y, meta = generate_observations(data_y, num_observations, do_add_noise)
    return [data_x, data_y, obs_y, meta]
