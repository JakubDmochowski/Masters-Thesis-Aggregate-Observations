import numpy as np
from models.gbm.aggregate_model import AggregateModel
from models.gbm.standard_model import StandardModel
from data.dataset import Dataset
from data.tabular.criteo import retrieve_data
from data.data_utils import observation_subset_for, split_data
import torch
from plot_utils import plot_confusion_matrix

RANDOM_SEED = 2022

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# global variables
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1
TRAIN_PARAMS = {
    "num_boost_round": 100,
    "early_stopping_rounds": 10,
}

MODEL_TYPE = "gbm"
# LOAD_MODEL = f"{MODEL_TYPE}_2022-08-09"
MODEL_DATE = "2022-09-09"
MODEL_INDEX = "0"
LOAD = False
LOAD_MODEL = f"{MODEL_DATE}_{MODEL_INDEX}" if LOAD else None

STANDARD_MODEL_KEY = "std"
AGGREGATE_MODEL_KEY = "agg"

data_x = None
data_y = None
expected_y = None
obs_y = None
meta = None
valFunc = None

data_x, data_y, obs_y, meta = retrieve_data()
expected_y = data_y

meta_train, meta_validate, meta_test = split_data(
    meta, test_split=TEST_SPLIT, validation_split=VALIDATION_SPLIT, random_state=RANDOM_SEED)
data_train = Dataset(data_x=data_x, data_y=data_y,
                     obs_y=obs_y, observations=meta_train)
data_test = Dataset(data_x=data_x, data_y=data_y,
                    obs_y=obs_y, observations=meta_test)
data_validate = Dataset(
    data_x=data_x, data_y=expected_y, obs_y=obs_y, observations=meta_validate)

aggregate_model = AggregateModel(train_params=TRAIN_PARAMS)
standard_model = StandardModel(train_params=TRAIN_PARAMS)
if LOAD_MODEL is not None:
    aggregate_model.load(MODEL_TYPE, LOAD_MODEL, AGGREGATE_MODEL_KEY)
    standard_model.load(MODEL_TYPE, LOAD_MODEL, STANDARD_MODEL_KEY)
else:
    aggregate_model.train(dataset=data_train, validate=data_validate)
    standard_model.train(dataset=data_train, validate=data_validate)
    aggregate_model.save(MODEL_TYPE, AGGREGATE_MODEL_KEY)
    standard_model.save(MODEL_TYPE, STANDARD_MODEL_KEY)

data_x_a, aggregate_predictions = aggregate_model.test(
    dataset=data_test)
data_x_s, standard_predictions = standard_model.test(
    dataset=data_test)

targets = observation_subset_for(data=expected_y, dataset=data_validate)
prediction_data = [
    {
        "label": 'aggregate model',
        "prediction_history": [torch.Tensor([[e, 1 - e] for e in aggregate_predictions])],
    },
    {
        "label": 'standard model',
        "prediction_history": [torch.Tensor([[e, 1 - e] for e in standard_predictions])],
    }
]
plot_confusion_matrix(prediction_data, targets)

input("Press Enter to continue...")
