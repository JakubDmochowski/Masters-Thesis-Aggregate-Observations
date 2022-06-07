import torch
from dataset import Dataset
from torch import nn
import numpy as np
from itertools import chain
from typing import Callable
from dataset import Observation
import torch.nn.functional as F


def length_to_range(lengths: list[int]):
    lengths = [0] + np.cumsum(lengths).tolist()
    return [range(a, b) for a, b in zip(lengths[:-1], lengths[1:])]


class AggregateLosses:
    def gaussian(self, entry_predictions: torch.tensor, observations: torch.tensor, lengths: list[int]):
        ranges = length_to_range(lengths)
        predictions = torch.stack(
            [entry_predictions[r].mean(axis=0) for r in ranges])
        return F.mse_loss(predictions, observations)


class AggregateModel:
    def __init__(self):
        self.output_dim = None
        self.input_dim = None
        self.model = None

    def getModelFor(self, dataset: Dataset):
        Dataset.validate(dataset)
        self.input_dim = len(dataset.data_x[0])
        self.output_dim = len(dataset.data_y[0])
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.output_dim),
        )

    def parameters(self):
        return self.model.parameters()

    def train(self, dataset: Dataset, optimizer, loss: Callable, batch_size: int) -> None:
        data_y_batch_indices = np.random.choice(
            len(dataset.observations), size=batch_size)
        observations_batch = np.array(
            dataset.observations).take(data_y_batch_indices)
        data_x_batch_indices = list(
            chain(*[obs.entries_indices for obs in observations_batch]))

        x_batch = dataset.data_x[data_x_batch_indices]
        y_batch = dataset.data_y[data_y_batch_indices]
        l_batch = [obs.length for obs in observations_batch]

        optimizer.zero_grad()
        l = loss(self.model(x_batch), y_batch, l_batch)
        l.backward()
        optimizer.step()
        return l

    def test(self, dataset: Dataset):
        data_x_indices = list(
            chain(*[obs.entries_indices for obs in dataset.observations]))
        x = dataset.data_x[data_x_indices]
        return [x, self.model(x)]
