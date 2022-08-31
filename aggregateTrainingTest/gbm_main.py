import numpy as np
from models.gbm.aggregate_model import AggregateModel
from models.gbm.standard_model import StandardModel
from data.dataset import Dataset
from data.tabular.criteo import retrieveData
from data.data_utils import observationSubsetFor, splitData
import torch
from plot_utils import plotConfusionMatrix

RANDOM_SEED = 2022

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# global variables
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1
VALIDATE_EVERY_K_ITERATIONS = 5
USE_TABULAR_DATA = True
LOSS = torch.nn.functional.mse_loss
USE_WEIGHT = True
CLASSIFICATION = USE_TABULAR_DATA

data_x = None
data_y = None
expected_y = None
obs_y = None
meta = None
valFunc = None

data_x, data_y, obs_y, meta = retrieveData()
expected_y = data_y

meta_train, meta_validate, meta_test = splitData(
    meta, test_split=TEST_SPLIT, validation_split=VALIDATION_SPLIT, random_state=RANDOM_SEED)
data_train = Dataset(data_x=data_x, data_y=data_y,
                     obs_y=obs_y, observations=meta_train)
data_test = Dataset(data_x=data_x, data_y=data_y,
                    obs_y=obs_y, observations=meta_test)
data_validate = Dataset(
    data_x=data_x, data_y=expected_y, obs_y=obs_y, observations=meta_test)

aggregate_model = AggregateModel()
standard_model = StandardModel()

data_train.useDevice(device)
data_validate.useDevice(device)
aggregate_model.train(dataset=data_train, validate=data_validate)
standard_model.train(dataset=data_train, validate=data_validate)

data_x_a, aggregate_predictions = aggregate_model.test(
    dataset=data_test)
data_x_s, standard_predictions = standard_model.test(
    dataset=data_test)

targets = observationSubsetFor(data=expected_y, dataset=data_validate)
prediction_data = [
    {
        "label": 'aggregate model',
        "prediction_history": [torch.Tensor([[e, 1-e] for e in aggregate_predictions])],
    },
    {
        "label": 'standard model',
        "prediction_history": [torch.Tensor([[e, 1-e] for e in standard_predictions])],
    }
]
plotConfusionMatrix(prediction_data, targets,
                    every=VALIDATE_EVERY_K_ITERATIONS)


input("Press Enter to continue...")
