from data.dataset import Dataset
from models.gbm.base_model import Model
import lightgbm as lgb
import numpy as np
from typing import Callable, Tuple


class AggregateModel(Model):

    def grad_hess_mean_gaussian(self, preds: np.ndarray, label: np.ndarray):
        k = preds.shape[0]
        y = label[0]
        grad = np.ones(k) / k * (preds.mean() - y)
        hess = np.ones(k) / k ** 2
        return grad, hess

    def aggregate_obj(self, grad_hess_func: Callable):
        def obj(preds: np.ndarray, train_data: lgb.Dataset) -> Tuple[np.ndarray, np.ndarray]:
            label = train_data.get_label()
            group = train_data.get_group()

            grad = np.zeros(len(preds))
            hess = np.zeros(len(preds))

            head, last = 0, 0
            for num_i in group.astype(int):
                head, last = last, last + num_i

                preds_i = preds[head:last]
                label_i = label[head:last]

                grad_i, hess_i = grad_hess_func(preds_i, label_i)

                grad[head:last] = grad_i
                hess[head:last] = hess_i

            return grad, hess

        return obj

    def train(self, dataset: Dataset, validate: Dataset) -> None:
        lgb_train = self.toLGBDataset(dataset)
        lgb_validate = self.toLGBDataset(validate)
        self.gbm = lgb.train(
            params=self.lgb_params,
            train_set=lgb_train,
            valid_sets=[lgb_validate],
            num_boost_round=self.train_params["num_boost_round"],
            fobj=self.aggregate_obj(self.grad_hess_mean_gaussian),
            callbacks=[lgb.early_stopping(
                self.train_params["early_stopping_rounds"])]
        )
