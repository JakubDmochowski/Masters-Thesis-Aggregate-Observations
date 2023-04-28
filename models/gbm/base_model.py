from data.dataset import Dataset
from itertools import chain
import lightgbm as lgb
import numpy as np
import csv
import os
from data.tabular.criteo import get_meta
from datetime import date
import re


class Model:
    def __init__(self, params: dict = {}, train_params: dict = {}, history: dict = {}):
        self.history = history
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
            "num_boost_round": 50000,
            "early_stopping_rounds": 100,
        }
        self.train_params = default_train_params | train_params

    @staticmethod
    def to_lgb_dataset(dataset: Dataset) -> lgb.Dataset:
        data_indices = list(
            chain(*[obs.entries_indices for obs in dataset.observations]))

        data = np.array(dataset.data_x[data_indices])
        labels = np.array(dataset.data_y[data_indices][:, 0])
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

    def save(self, model_type: str, model_key: str) -> None:
        savedir = f"models/{model_type}/saved"
        if not os.path.isdir(savedir):
            os.mkdir(savedir)
        # filename = f"{date.today()}"
        today = date.today()
        files = [f for f in os.listdir(
            savedir) if os.path.isfile(os.path.join(savedir, f))]
        regex = f"^{today}_([0-9]+)_[A-Z]+$"
        idx = [int(re.search(regex, f).group(1)) for f in files if re.search(
            regex, f)]
        index = max([*idx, 0])
        basename = f"{today}_{index}"
        filename = f"{basename}_{model_key}"
        self.gbm.save_model(f"{savedir}/{filename}",
                            num_iteration=self.gbm.best_iteration)

        meta_filename = f"{basename}.meta"
        if not os.path.exists(meta_filename):
            meta = self.lgb_params | self.train_params | get_meta()
            meta_file = open(f"{savedir}/{meta_filename}", "w", newline='')
            meta_file_writer = csv.writer(meta_file)
            for entry in meta:
                meta_file_writer.writerow([entry, meta[entry]])
            meta_file.close()

    def load(self, model_type: str, load_filename, model_key: str) -> None:
        savedir = f"models/{model_type}/saved"
        filename = savedir + "/" + load_filename + f"_{model_key}"
        self.gbm = lgb.Booster(model_file=filename)
