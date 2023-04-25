import math
import numpy as np
from models.neural.aggregate_model import AggregateModel
from models.neural.standard_model import StandardModel
from data.dataset import Dataset
from data.tabular.breast_cancer_1 import retrieve_data, get_weights
from data.synthetic import generate_data
from data.data_utils import generate_values, observation_subset_for, split_data
import torch
from torch import optim
from tqdm import trange, tqdm
from plot_utils import plot_xy, plot_losses, plot_auc, plot_precision, plot_recall, plot_confusion_matrix

RANDOM_SEED = 2022

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# global variables

BATCH_SIZE = 32
NUM_ITERS = 1000
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1
VALIDATE_EVERY_K_ITERATIONS = 5
USE_TABULAR_DATA = True
LEARNING_RATE = 0.001
LOSS = torch.nn.functional.mse_loss
USE_WEIGHT = True
if USE_TABULAR_DATA is True:
    if USE_WEIGHT is True:
        LOSS = torch.nn.BCELoss(weight=torch.Tensor(get_weights()))
    else:
        LOSS = torch.nn.BCELoss()
CLASSIFICATION = USE_TABULAR_DATA

# synthetic-data-only variables
NUM_ENTRIES = 1000
GROUP_SIZE = 5
NUM_OBSERVATIONS = math.ceil(NUM_ENTRIES/GROUP_SIZE)
NUM_DIMENSIONS = 1
ADD_NOISE = False


data_x = None
data_y = None
expected_y = None
obs_y = None
meta = None
valFunc = None

def aggregate_sum(z: torch.tensor):
        return z.sum(axis=0)


if USE_TABULAR_DATA:
    data_x, data_y, obs_y, meta = retrieve_data(
        group_size=GROUP_SIZE)
    expected_y = data_y
else:
    def val_func_1(x: list[float]) -> np.ndarray:
        # returns array of y values. each y value is a function f(y_i) = (x_1, ..., x_n)
        val = [np.array([dim * dim for dim in x]).sum()]
        return val
    def val_func_2(x: list[float]) -> np.ndarray:
        # returns array of y values. each y value is a function f(y_i) = (x_1, ..., x_n)
        val = np.array([dim % 4 - (dim / 4) for dim in x])
        return val

    val_func = val_func_1

    def aggregate_mean(z: torch.tensor):
        return z.mean(axis=0)

    data_x, data_y, obs_y, meta = generate_data(entry_no=NUM_ENTRIES,
                                                dim_no=NUM_DIMENSIONS,
                                                num_observations=NUM_OBSERVATIONS,
                                                do_add_noise=ADD_NOISE,
                                                value_func=val_func,
                                                aggregate=aggregate_sum)
    expected_y = generate_values(
        data_x=data_x, value_func=val_func)

meta_train, meta_validate, meta_test = split_data(
    meta, test_split=TEST_SPLIT, validation_split=VALIDATION_SPLIT, random_state=RANDOM_SEED)
data_train = Dataset(data_x=data_x, data_y=data_y,
                     obs_y=obs_y, observations=meta_train)
data_test = Dataset(data_x=data_x, data_y=data_y,
                    obs_y=obs_y, observations=meta_test)
data_validate = Dataset(
    data_x=data_x, data_y=expected_y, obs_y=obs_y, observations=meta_validate)

loss_history = []
aggregate_model = AggregateModel(classification=CLASSIFICATION, aggregate_by=aggregate_sum)
aggregate_model.get_model_for(data_train)
standard_model = StandardModel(classification=CLASSIFICATION)
standard_model.get_model_for(data_train)

aggregate_prediction_history = []
standard_prediction_history = []

AGGREGATE_BATCH = math.ceil(BATCH_SIZE / (NUM_ENTRIES * (1 - VALIDATION_SPLIT - TEST_SPLIT)) / NUM_OBSERVATIONS)
pbar = trange(NUM_ITERS)
for iterIndex in pbar:
    aggregate_loss = aggregate_model.train(dataset=data_train,
                                           optimizer=optim.Adam(
                                               aggregate_model.parameters(), lr=LEARNING_RATE),
                                           loss=LOSS,
                                           batch_size=BATCH_SIZE)
    pbar.set_description(desc=f"aggregate_loss = {aggregate_loss}")
    standard_loss = standard_model.train(dataset=data_train,
                                         optimizer=optim.Adam(
                                             standard_model.parameters(), lr=LEARNING_RATE),
                                         loss=LOSS,
                                         batch_size=BATCH_SIZE)
    loss_history.append(
        {'metoda Zhanga': aggregate_loss, 'metoda standardowa': standard_loss})

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

plot_losses(loss_history)
if USE_TABULAR_DATA is False:
    series = [
        # {
        #     "label": "Oczekiwany wynik",
        #     "marker": "o",
        #     "data_x": data_x,
        #     "data_y": expected_y
        # },
        {
            "label": "metoda Zhanga",
            "marker": "^",
            "data_x": data_x_a,
            "data_y": aggregate_predictions
        },
        {
            "label": "metoda standardowa",
            "marker": "o",
            "data_x": data_x_s,
            "data_y": standard_predictions
        },
    ]
    plot_xy(data_x=data_x, expected_y=expected_y,
            series=series, value_func=val_func, dim=0)
    if NUM_DIMENSIONS > 1:
        plot_xy(data_x=data_x, expected_y=expected_y,
                series=series, value_func=val_func, dim=1)
    std_targets = np.array(list(map(lambda x: val_func(x), data_x_s))).reshape(-1)
    agg_targets = np.array(list(map(lambda x: val_func(x), data_x_a))).reshape(-1)
    std_mse = np.square(np.array(standard_predictions).reshape(-1) - std_targets)
    agg_mse = np.square(np.array(aggregate_predictions).reshape(-1) - agg_targets)
    print("mean squared error")
    print(f"std: {LOSS(torch.tensor(std_targets), torch.tensor(standard_predictions))}")
    print(f"zhang: {LOSS(torch.tensor(agg_targets), torch.tensor(aggregate_predictions))}")
    print("standard deviations")
    print(f"std: {np.std(std_mse)}")
    print(f"zhang: {np.std(agg_mse)}")
else:
    targets = observation_subset_for(data=expected_y, dataset=data_validate)
    prediction_data = [
        {
            "label": 'uczenie Zhanga',
            "prediction_history": aggregate_prediction_history,
        },
        {
            "label": 'uczenie standardowe',
            "prediction_history": standard_prediction_history,
        }
    ]
    test_prediction_data = [
        {
            "label": 'uczenie Zhanga',
            "prediction_history": [aggregate_predictions],
        },
        {
            "label": 'uczenie standardowe',
            "prediction_history": [standard_predictions],
        }
    ]
    test_targets = observation_subset_for(data=expected_y, dataset=data_test)
    plot_auc(prediction_data, targets,
             every=VALIDATE_EVERY_K_ITERATIONS)
    plot_precision(prediction_data, targets,
                   every=VALIDATE_EVERY_K_ITERATIONS)
    plot_recall(prediction_data, targets,
                every=VALIDATE_EVERY_K_ITERATIONS)
    plot_confusion_matrix(prediction_data, targets)
    plot_confusion_matrix(test_prediction_data, test_targets)

input("Press Enter to continue...")
