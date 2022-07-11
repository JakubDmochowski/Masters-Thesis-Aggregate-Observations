from XBNet.models import XBNETClassifier, XBNETRegressor
from data.dataset import Dataset
from itertools import chain


class Model:
    def __init__(self, layers_raw: list[dict], classification: bool = False):
        self.output_nodes = None
        self.input_nodes = None
        self.model = None
        self.classification = classification
        self.layers_raw = layers_raw

    def parameters(self):
        return self.model.parameters()

    def getModelFor(self, dataset: Dataset) -> None:
        Dataset.validate(dataset)
        if self.classification:
            self.model = XBNETClassifier(dataset, layers_raw=self.layers_raw)
        # else:
        #     self.model = XBNETRegressor(dataset, layers_raw=self.layers_raw)

    def test(self, dataset: Dataset):
        data_x_indices = list(
            chain(*[obs.entries_indices for obs in dataset.observations]))
        x = dataset.data_x[data_x_indices]
        return [x, self.model(x)]
