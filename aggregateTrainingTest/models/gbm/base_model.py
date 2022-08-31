from data.dataset import Dataset
from itertools import chain
import lightgbm as lgb
import numpy as np


class Model:
    def __init__(self, params: dict = {}, train_params: dict = {}):
        self.gbm = None
        default_lgb_params = {
            "objective": "binary",
            "learning_rate": 0.01,
            "boosting_type": "gbdt",
            "random_state": 42,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.8,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": 2,
            "metrics": "binary_logloss"
        }

        self.lgb_params = default_lgb_params | params
        default_train_params = {
            "num_boost_round": 500,
            "early_stopping_rounds": 100,
        }
        self.train_params = default_train_params | train_params

    def toLGBDataset(self, dataset: Dataset) -> lgb.Dataset:
        data_indices = list(
            chain(*[obs.entries_indices for obs in dataset.observations]))

        data = dataset.data_x[data_indices].numpy()
        labels = np.ascontiguousarray(
            dataset.data_y[data_indices][:, 0].numpy())
        lengths = np.array([obs.length for obs in dataset.observations])
        return lgb.Dataset(data=data, label=labels, group=lengths)

    def parameters(self):
        return self.params

    def model(self, x):
        return self.gbm.predict(x)

    def test(self, dataset: Dataset):
        data_x_indices = list(
            chain(*[obs.entries_indices for obs in dataset.observations]))
        x = dataset.data_x[data_x_indices]
        return [x, self.model(x)]
