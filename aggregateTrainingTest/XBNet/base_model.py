from XBNet.models import XBNETClassifier, XBNETRegressor
from data.dataset import Dataset
from itertools import chain


class Model:
    def __init__(self, classification: bool = False):
        self.output_dim = None
        self.input_dim = None
        self.model = None
        self.classification = classification

    def parameters(self):
        return self.model.parameters()

    def getModelFor(self, dataset: Dataset) -> None:
        Dataset.validate(dataset)
        if self.classification:
            self.model = XBNETClassifier(dataset)
        else:
            self.model = XBNETRegressor(dataset)

    def test(self, dataset: Dataset):
        data_x_indices = list(
            chain(*[obs.entries_indices for obs in dataset.observations]))
        x = dataset.data_x[data_x_indices]
        return [x, self.model(x)]
