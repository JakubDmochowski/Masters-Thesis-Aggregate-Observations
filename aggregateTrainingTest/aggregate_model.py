import torch
from dataset import Dataset
import numpy as np
from itertools import chain
from typing import Callable
from dataset import Observation
import torch.nn.functional as F
from aggregate_utils import length_to_range
from base_model import Model


class AggregateLosses:
    def gaussian(self, entry_predictions: torch.tensor, observations: torch.tensor, lengths: list[int]):
        ranges = length_to_range(lengths)
        predictions = torch.stack(
            [entry_predictions[r].mean(axis=0) for r in ranges])
        return F.mse_loss(predictions, observations) * np.array(lengths).sum()


class AggregateModel(Model):
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
        l = loss(self.model(x_batch), y_batch, l_batch)
        l.backward()
        optimizer.step()
        return l
