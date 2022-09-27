import torch
from data.dataset import Dataset
import numpy as np
from itertools import chain
from typing import Callable
from aggregate_utils import length_to_range
from models.neural.base_model import Model
from typing import Callable


class AggregateModel(Model):
    def applyAggregateLoss(self, loss: Callable, entry_predictions: torch.tensor, observations: torch.tensor, lengths: list[int]):
        ranges = length_to_range(lengths)
        predictions = torch.stack(
            [entry_predictions[r].mean(axis=0) for r in ranges])
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
        l = self.applyAggregateLoss(
            loss, self.model(x_batch), y_batch, l_batch)
        l.backward()
        optimizer.step()
        return l
