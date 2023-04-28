import numpy as np
import torch

from data.dataset import Dataset
from models.gbm.base_model import Model
import lightgbm as lgb
from sklearn import metrics


class StandardModel(Model):
    def train(self, dataset: Dataset, test: Dataset) -> None:
        lgb_train = self.to_lgb_dataset(dataset)
        lgb_test = self.to_lgb_dataset(test)

        def accuracy(predictions: np.ndarray, eval_data: lgb.Dataset) -> (str, float, bool):
            input = torch.tensor([1 if x >= 0.5 else 0 for x in predictions], dtype=torch.float64)
            target = torch.tensor(eval_data.get_label(), dtype=torch.float64)
            return 'accuracy', metrics.accuracy_score(target, input), True

        def binary_logloss(predictions: np.ndarray, eval_data: lgb.Dataset) -> (str, float, bool):
            # predictions shaped [n_samples, n_classes]
            loss = torch.nn.BCELoss()
            input = torch.tensor([min(1, max(0, entry)) for entry in predictions], dtype=torch.float64)
            target = torch.tensor(eval_data.get_label(), dtype=torch.float64)
            return 'binary_logloss', loss(input, target), False

        def auc(predictions: np.ndarray, eval_data: lgb.Dataset) -> (str, float, bool):
            # predictions shaped [n_samples, n_classes]
            input = np.array([min(1, max(0, entry)) for entry in predictions])
            target = np.array(eval_data.get_label())
            return 'auc', metrics.roc_auc_score(target, input), True

        self.gbm = lgb.train(
            params=self.lgb_params,
            train_set=lgb_train,
            valid_sets=[lgb_test],
            valid_names=['test'],
            feval=[binary_logloss, auc, accuracy],
            num_boost_round=self.train_params["num_boost_round"],
            callbacks=[lgb.record_evaluation(self.history)],
            # callbacks=[lgb.early_stopping(
            #     self.train_params["early_stopping_rounds"], first_metric_only=True), lgb.record_evaluation(self.history)]
        )
