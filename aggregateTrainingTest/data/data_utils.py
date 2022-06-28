import torch
from dataset import Dataset
from typing import Callable
import numpy as np
from itertools import chain
from sklearn.model_selection import train_test_split


def observationSubsetFor(data: torch.tensor, dataset: Dataset):
    data_indices = list(
        chain(*[obs.entries_indices for obs in dataset.observations]))
    return data[data_indices]


def generateValues(data_x: torch.tensor, value_func: Callable) -> np.ndarray:
    # returned data_y is a tensor shaped (entries, values)
    return torch.tensor(np.array([value_func(x) for x in data_x.numpy()])).float()


def splitData(meta, test_split, validation_split, random_state):
    vt_size = validation_split + test_split
    meta_train, meta_other = train_test_split(
        meta, test_size=vt_size, random_state=random_state)

    meta_validation, meta_test = train_test_split(
        meta_other, test_size=validation_split / vt_size, random_state=random_state)
    return meta_train, meta_validation, meta_test
