import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from aggregate_model import AggregateLosses, AggregateModel
from standard_model import StandardLosses, StandardModel
from dataset import Dataset
from synthetic import generateData, generateObservations, generateValues, getExpectedValues
from quality_measures import RegressionQualityMeasure
from sklearn.model_selection import train_test_split
import torch
from torch import optim
from tqdm import trange
from typing import Callable

np.random.seed(2020)

NUM_ENTRIES = 300
NUM_DIMENSIONS = 1
NUM_OBSERVATIONS = 120
TEST_SPLIT = 0.2
BATCH_SIZE = 32
NUM_ITERS = 1000
ADD_NOISE = False


def valFunc(x: list[float]) -> np.ndarray:
    # returns array of y values. each y value is a function f(y_i) = (x_1, ..., x_n)
    val = np.array([np.sum(dim*dim) for dim in x])
    return val


AGGREGATE_MODEL_TRAIN_LOSS = AggregateLosses().gaussian
STANDARD_MODEL_TRAIN_LOSS = StandardLosses().mse_loss
QUALITY_MEASURE = RegressionQualityMeasure.MeanSquaredError

data_x = generateData(entry_no=NUM_ENTRIES,
                      dim_no=NUM_DIMENSIONS)
expected_y = generateValues(
    data_x=data_x, value_func=valFunc)
data_y, meta = generateObservations(data_x=data_x,
                                    num_observations=NUM_OBSERVATIONS,
                                    add_noise=ADD_NOISE,
                                    value_func=valFunc)
meta_train, meta_test = train_test_split(meta, test_size=TEST_SPLIT)
data_train = Dataset(data_x=data_x, data_y=data_y, observations=meta_train)
data_test = Dataset(data_x=data_x, data_y=data_y, observations=meta_test)
data_validate = Dataset(
    data_x=data_x, data_y=expected_y, observations=meta_test)


def plotLosses(loss_history: list[list[float]]):
    fig, ax = plt.subplots(figsize=(8, 8))
    x = range(0, len(loss_history))
    aggregate = [losses[0].detach().numpy() for losses in loss_history]
    standard = [losses[1].detach().numpy() for losses in loss_history]
    ax.plot(x, aggregate, label="aggregate")
    ax.plot(x, standard, label="standard")
    ax.set_xlabel('iteration')
    ax.set_ylabel('loss')
    ax.legend()
    fig.show()


def preparePlot(data_x: torch.tensor, expected_y: torch.tensor, valueFunc: Callable):
    fig, ax = plt.subplots(figsize=(8, 8))
    data_x = np.array([x[0] for x in data_x.numpy()])
    expected_y = np.array([y[0] for y in expected_y.numpy()])
    x_min, x_max = [np.min(data_x), np.max(data_x)]
    x_lin = np.linspace(x_min, x_max, 500)
    y_lin = list(map(lambda x: valueFunc([x]), x_lin))
    ax.plot(x_lin, y_lin, color="k",
            linewidth=3, label="valueFunc")
    ax.scatter(data_x, expected_y, color="r", label="expectation")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return fig, ax


def plot(ax, iters, data_x, predictions, label, marker):
    ax.scatter(data_x, predictions,
               label=f"{label}_prediction_{iters}", marker=marker)


fig, ax = preparePlot(data_x=data_x, expected_y=expected_y, valueFunc=valFunc)

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
    data_x, aggregate_predictions = aggregate_model.test(dataset=data_test)
    _, standard_predictions = standard_model.test(dataset=data_test)
# print(f"predictions: {predictions[range(0,5)]}")
# print(f"expectations: {expectations[range(0,5)]}")
aggregate_quality = QUALITY_MEASURE(expectations, aggregate_predictions)
standard_quality = QUALITY_MEASURE(expectations, standard_predictions)

plot(ax, NUM_ITERS, data_x, aggregate_predictions, label="aggregate", marker="^")
plot(ax, NUM_ITERS, data_x, standard_predictions, label="standard", marker="o")

plotLosses(loss_history)

ax.legend()
fig.show()
input("Press Enter to continue...")
