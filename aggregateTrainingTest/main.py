import numpy as np
from aggregate_model import AggregateModel
from standard_model import StandardModel
from dataset import Dataset
from data.tabular import retrieveData
from data.synthetic import generateData
from data.data_utils import generateValues, observationSubsetFor, splitData
import torch
from torch import optim
from tqdm import trange
from plot_utils import plotXY, plotLosses, plotROC, plotAUC
import torch.nn.functional as F
from sklearn import svm

RANDOM_SEED = 2022

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# global variables
NUM_OBSERVATIONS = 50
BATCH_SIZE = 32
NUM_ITERS = 800
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1
VALIDATE_EVERY_K_ITERATIONS = 5
USE_TABULAR_DATA = True
LOSS = torch.nn.functional.mse_loss
if USE_TABULAR_DATA is True:
    LOSS = torch.nn.BCELoss()
CLASSIFICATION = USE_TABULAR_DATA

# synthetic-data-only variables
NUM_ENTRIES = 300
NUM_DIMENSIONS = 2
ADD_NOISE = False


data_x = None
data_y = None
expected_y = None
obs_y = None
meta = None
valFunc = None

if USE_TABULAR_DATA:
    data_x, data_y, obs_y, meta = retrieveData(
        num_observations=NUM_OBSERVATIONS)
    expected_y = data_y
else:
    def valFunc(x: list[float]) -> np.ndarray:
        # returns array of y values. each y value is a function f(y_i) = (x_1, ..., x_n)
        val = np.array([np.sum(dim*dim) for dim in x])
        return val

    data_x, data_y, obs_y, meta = generateData(entry_no=NUM_ENTRIES,
                                               dim_no=NUM_DIMENSIONS,
                                               num_observations=NUM_OBSERVATIONS,
                                               add_noise=ADD_NOISE,
                                               value_func=valFunc)
    expected_y = generateValues(
        data_x=data_x, value_func=valFunc)

meta_train, meta_validate, meta_test = splitData(
    meta, test_split=TEST_SPLIT, validation_split=VALIDATION_SPLIT, random_state=RANDOM_SEED)
data_train = Dataset(data_x=data_x, data_y=data_y,
                     obs_y=obs_y, observations=meta_train)
data_test = Dataset(data_x=data_x, data_y=data_y,
                    obs_y=obs_y, observations=meta_test)
data_validate = Dataset(
    data_x=data_x, data_y=expected_y, obs_y=obs_y, observations=meta_test)

loss_history = []
aggregate_model = AggregateModel(classification=CLASSIFICATION)
aggregate_model.getModelFor(data_train)
standard_model = StandardModel(classification=CLASSIFICATION)
standard_model.getModelFor(data_train)

aggregate_prediction_history = []
standard_prediction_history = []

for iterIndex in trange(NUM_ITERS):
    aggregate_loss = aggregate_model.train(dataset=data_train,
                                           optimizer=optim.Adam(
                                               aggregate_model.parameters()),
                                           loss=LOSS,
                                           batch_size=BATCH_SIZE)
    standard_loss = standard_model.train(dataset=data_train,
                                         optimizer=optim.Adam(
                                             standard_model.parameters()),
                                         loss=LOSS,
                                         batch_size=BATCH_SIZE)
    loss_history.append([aggregate_loss, standard_loss])

    if not iterIndex % VALIDATE_EVERY_K_ITERATIONS:
        with torch.no_grad():
            data_x_a, aggregate_predictions = aggregate_model.test(
                dataset=data_validate)
            data_x_s, standard_predictions = standard_model.test(
                dataset=data_validate)
            aggregate_prediction_history.append(aggregate_predictions)
            standard_prediction_history.append(standard_predictions)

with torch.no_grad():
    data_x_a, aggregate_predictions = aggregate_model.test(
        dataset=data_test)
    data_x_s, standard_predictions = standard_model.test(
        dataset=data_test)

plotLosses(loss_history)
if USE_TABULAR_DATA is False:
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
    plotXY(data_x=data_x, expected_y=expected_y,
           series=series, value_func=valFunc)
else:
    # SVM baseline algorithm
    # x_train = observationSubsetFor(data=data_x, dataset=data_train)
    # y_train = observationSubsetFor(data=expected_y, dataset=data_train)
    # x_test = observationSubsetFor(data=data_x, dataset=data_test)
    # y_test = observationSubsetFor(data=expected_y, dataset=data_test)
    # clf = svm.SVC(kernel='linear', degree=2)
    # clf.fit(x_train, y_train[:, 0])
    # y_pred = clf.predict(x_test)
    # plotROC(y_test[:, 0], y_pred, "SVM ROC")
    # SVM end
    # plotROC(targets, aggregate_predictions, "aggregate ROC")
    # plotROC(targets, standard_predictions, "standard ROC")
    targets = observationSubsetFor(data=expected_y, dataset=data_validate)
    auc_data = [
        {
            "label": 'aggregate model',
            "prediction_history": aggregate_prediction_history,
        },
        {
            "label": 'standard model',
            "prediction_history": standard_prediction_history,
        }
    ]
    plotAUC(auc_data, targets,
            every=VALIDATE_EVERY_K_ITERATIONS)

input("Press Enter to continue...")
