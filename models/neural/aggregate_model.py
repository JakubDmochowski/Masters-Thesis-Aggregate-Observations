import torch
from data.dataset import Dataset
import numpy as np
from itertools import chain
from data.aggregate_utils import length_to_range
from models.neural.base_model import Model
from typing import Callable


def aggregate_mean(z: torch.tensor):
    return z.mean(axis=0)


def default_aggregate_by(z: torch.tensor):
    return aggregate_mean(z)


class AggregateModel(Model):
    def __init__(self, classification: bool = False, aggregate_by: Callable = default_aggregate_by):
        super().__init__(classification=classification)
        self.aggregate_by = aggregate_by

    def apply_aggregate_loss(self, loss: Callable, entry_predictions: torch.tensor, observations: torch.tensor,
                             lengths: list[int]):
        ranges = length_to_range(lengths)
        predictions = torch.stack(
            [self.aggregate_by(entry_predictions[r]) for r in ranges])
        return loss(predictions, observations) * (np.array(lengths).sum() / len(lengths))

    def train(self, dataset: Dataset, optimizer, loss: Callable, batch_size: int) -> None:
        data_y_batch_indices = np.random.choice(
            len(dataset.observations), size=batch_size)
        observations_batch = np.array(
            dataset.observations).take(data_y_batch_indices)
        data_x_batch_indices = list(
            chain(*[obs.entries_indices for obs in observations_batch]))
        obs_y_batch_indices = [
            obs.value_vec_index for obs in observations_batch]

        x_batch = dataset.data_x[data_x_batch_indices]
        y_batch = dataset.obs_y[obs_y_batch_indices]
        l_batch = [obs.length for obs in observations_batch]

        optimizer.zero_grad()
        l = self.apply_aggregate_loss(
            loss, self.model(x_batch), y_batch, l_batch)
        l.backward()
        optimizer.step()
        return l
