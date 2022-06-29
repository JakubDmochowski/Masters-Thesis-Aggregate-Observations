import torch
from data.dataset import Dataset
import numpy as np
from itertools import chain
from typing import Callable
from classical.base_model import Model


def length_to_range(lengths: list[int]):
    lengths = [0] + np.cumsum(lengths).tolist()
    return [range(a, b) for a, b in zip(lengths[:-1], lengths[1:])]


class StandardModel(Model):
    def train(self, dataset: Dataset, optimizer, loss: Callable, batch_size: int) -> None:
        data_y_batch_indices = np.random.choice(
            len(dataset.observations), size=batch_size)
        observations_batch = np.array(
            dataset.observations).take(data_y_batch_indices)
        data_x_batch_indices = list(
            chain(*[obs.entries_indices for obs in observations_batch]))

        x_batch = dataset.data_x[data_x_batch_indices]
        y_batch = dataset.data_y[data_x_batch_indices]

        optimizer.zero_grad()
        l = loss(self.model(x_batch), y_batch)
        l.backward()
        optimizer.step()
        return l
