import numpy as np
from aggregate_model import AggregateLosses, AggregateModel
from standard_model import StandardLosses, StandardModel
from dataset import Dataset
from synthetic import generateData, generateObservations, generateValues, getExpectedValues
from sklearn.model_selection import train_test_split
import torch
from torch import optim
from tqdm import trange
from plot_utils import plotXY, plotLosses

np.random.seed(2020)

NUM_ENTRIES = 300
NUM_DIMENSIONS = 1
NUM_OBSERVATIONS = 120
TEST_SPLIT = 0.2
BATCH_SIZE = 32
NUM_ITERS = 400
ADD_NOISE = False


def valFunc(x: list[float]) -> np.ndarray:
    # returns array of y values. each y value is a function f(y_i) = (x_1, ..., x_n)
    val = np.array([np.sum(dim*dim) for dim in x])
    return val


AGGREGATE_MODEL_TRAIN_LOSS = AggregateLosses().gaussian
STANDARD_MODEL_TRAIN_LOSS = StandardLosses().mse_loss

data_x = generateData(entry_no=NUM_ENTRIES,
                      dim_no=NUM_DIMENSIONS)
expected_y = generateValues(
    data_x=data_x, value_func=valFunc)
data_y, obs_y, meta = generateObservations(data_x=data_x,
                                           num_observations=NUM_OBSERVATIONS,
                                           add_noise=ADD_NOISE,
                                           value_func=valFunc)
meta_train, meta_test = train_test_split(meta, test_size=TEST_SPLIT)
data_train = Dataset(data_x=data_x, data_y=data_y,
                     obs_y=obs_y, observations=meta_train)
data_test = Dataset(data_x=data_x, data_y=data_y,
                    obs_y=obs_y, observations=meta_test)
data_validate = Dataset(
    data_x=data_x, data_y=expected_y, obs_y=obs_y, observations=meta_test)

loss_history = []
aggregate_model = AggregateModel()
aggregate_model.getModelFor(data_train)
standard_model = StandardModel()
standard_model.getModelFor(data_train)
for _ in trange(NUM_ITERS):
    aggregate_loss = aggregate_model.train(dataset=data_train,
                                           optimizer=optim.Adam(
                                               aggregate_model.parameters()),
                                           loss=AGGREGATE_MODEL_TRAIN_LOSS,
                                           batch_size=BATCH_SIZE)
    standard_loss = standard_model.train(dataset=data_train,
                                         optimizer=optim.Adam(
                                             standard_model.parameters()),
                                         loss=STANDARD_MODEL_TRAIN_LOSS,
                                         batch_size=BATCH_SIZE)
    loss_history.append([aggregate_loss, standard_loss])
expectations = getExpectedValues(expected_y=expected_y, dataset=data_test)
aggregate_predictions = None
standard_predictions = None

with torch.no_grad():
    data_x_a, aggregate_predictions = aggregate_model.test(dataset=data_test)
    data_x_s, standard_predictions = standard_model.test(dataset=data_test)

series = [
    {
        "label": "expected",
        "marker": "o",
        "data_x": data_x,
        "data_y": expected_y
    },
    {
        "label": "aggregate_prediction",
        "marker": "^",
        "data_x": data_x_a,
        "data_y": aggregate_predictions
    },
    {
        "label": "standard_prediction",
        "marker": "o",
        "data_x": data_x_s,
        "data_y": standard_predictions
    },
]
plotXY(data_x=data_x, expected_y=expected_y, valueFunc=valFunc, series=series)
plotLosses(loss_history)
input("Press Enter to continue...")
