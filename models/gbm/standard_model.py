from data.dataset import Dataset
from models.gbm.base_model import Model
import lightgbm as lgb


class StandardModel(Model):
    def train(self, dataset: Dataset, validate: Dataset) -> None:
        lgb_train = self.toLGBDataset(dataset)
        lgb_validate = self.toLGBDataset(validate)
        self.gbm = lgb.train(
            params=self.lgb_params,
            train_set=lgb_train,
            valid_sets=[lgb_validate],
            num_boost_round=self.train_params["num_boost_round"],
            callbacks=[lgb.early_stopping(
                self.train_params["early_stopping_rounds"])]
        )
