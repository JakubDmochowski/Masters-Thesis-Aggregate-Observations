import torch
from data.dataset import Dataset
import numpy as np
from itertools import chain
from data.aggregate_utils import length_to_range
from models.neural.base_model import Model
from typing import Callable
import math


def aggregate_mean(z: torch.tensor):
    return z.mean(axis=0)


def default_aggregate_by(z: torch.tensor):
    return aggregate_mean(z)


def poi_t(ps: torch.Tensor, i: int):
    div = torch.pow(ps/(1-ps), i)
    return torch.sum(div)

def poi_bin(ps: torch.Tensor, k: int):
    ps = torch.clamp(ps, 10e-7, 0.999999)
    if k==0:
        return torch.prod((1-ps))
    else:
        out = torch.zeros(1)
        for i in range(1,k+1):
            out += math.pow(-1,i-1)*poi_bin(ps, k-i)*poi_t(ps, i)
        return out/k

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

    def apply_aggregate_poi_bin_loss(self, entry_predictions: torch.tensor, observations: torch.tensor, lengths: list[int]):        
        ranges = length_to_range(lengths)
        predictions = torch.stack([entry_predictions[r] for r in ranges],dim=0).squeeze()
        
        n = predictions.shape[0]
        out = torch.zeros(1)
        y=observations.int()
        n_classes = entry_predictions.shape[1]

        for i in range(n):
            for j in range(n_classes):
                v = poi_bin(predictions[i,:, j], y[i, j])    
                v = torch.clamp(v, 10e-7,10e+7)
                out -= torch.log(v)
        return out/n


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
        l = self.apply_aggregate_poi_bin_loss(self.model(x_batch), y_batch, l_batch)
        l.backward()
        optimizer.step()
        return l
