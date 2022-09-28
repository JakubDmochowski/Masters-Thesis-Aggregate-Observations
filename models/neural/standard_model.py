from data.dataset import Dataset
import numpy as np
from itertools import chain
from typing import Callable
from models.neural.base_model import Model


class StandardModel(Model):
    def train(self, dataset: Dataset, optimizer, loss: Callable, batch_size: int) -> None:
        """

        :rtype: object
        """
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
