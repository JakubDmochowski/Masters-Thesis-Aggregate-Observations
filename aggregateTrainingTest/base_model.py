from dataset import Dataset
from itertools import chain
from torch import nn


class Model:
    def __init__(self):
        self.output_dim = None
        self.input_dim = None
        self.model = None

    def parameters(self):
        return self.model.parameters()

    def getModelFor(self, dataset: Dataset) -> None:
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

    def test(self, dataset: Dataset):
        data_x_indices = list(
            chain(*[obs.entries_indices for obs in dataset.observations]))
        x = dataset.data_x[data_x_indices]
        return [x, self.model(x)]
