from data.dataset import Dataset
from models.gbm.base_model import Model
import lightgbm as lgb
import numpy as np
from typing import Callable, Tuple


class AggregateModel(Model):

    @staticmethod
    def grad_hess_mean_gaussian(predictions: np.ndarray, labels: np.ndarray):
        k = predictions.shape[0]
        y = labels[0]
        grad = np.ones(k) / k * (predictions.mean() - y)
        hess = np.ones(k) / k ** 2
        return grad, hess

    @staticmethod
    def aggregate_obj(grad_hess_func: Callable):
        def obj(predictions: np.ndarray, train_data: lgb.Dataset) -> Tuple[np.ndarray, np.ndarray]:
            labels = train_data.get_label()
            group = train_data.get_group()

            grad = np.zeros(len(predictions))
            hess = np.zeros(len(predictions))

            head, last = 0, 0
            for num_i in group.astype(int):
                head, last = last, last + num_i

                predictions_i = predictions[head:last]
                labels_i = labels[head:last]

                grad_i, hess_i = grad_hess_func(predictions_i, labels_i)

                grad[head:last] = grad_i
                hess[head:last] = hess_i

            return grad, hess

        return obj

    def train(self, dataset: Dataset, validate: Dataset) -> None:
        lgb_train = self.to_lgb_dataset(dataset)
        lgb_validate = self.to_lgb_dataset(validate)
        self.gbm = lgb.train(
            params=self.lgb_params,
            train_set=lgb_train,
            valid_sets=[lgb_validate],
            num_boost_round=self.train_params["num_boost_round"],
            fobj=self.aggregate_obj(self.grad_hess_mean_gaussian),
            callbacks=[lgb.early_stopping(
                self.train_params["early_stopping_rounds"])],
            init_model=self.gbm
        )
