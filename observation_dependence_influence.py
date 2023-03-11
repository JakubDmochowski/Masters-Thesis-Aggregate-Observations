import math
import numpy as np
from models.neural.aggregate_model import AggregateModel
from models.neural.standard_model import StandardModel
from data.dataset import Dataset
from data.tabular.breast_cancer_1 import retrieve_data, get_weights
from data.synthetic import generate_data, generate_dependent_observations, aggregate_mean
from data.data_utils import generate_values, observation_subset_for, split_data
import matplotlib.pyplot as plt
import torch
from torch import optim
from tqdm import trange
from plot_utils import plot_xy, plot_losses, plot_auc, plot_precision, plot_recall, plot_confusion_matrix, plot_xy3d

RANDOM_SEED = 2022

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

SHOW_INDIVIDUAL_PLOTS=False

REPEAT_TIMES = 10
DIMS = range(2,20)
NUM_ENTRIES = 1000

def aggregate_mean(entries: torch.tensor):
    return entries.mean(axis=0)

repeat_results = np.ndarray(shape=(REPEAT_TIMES, len(DIMS), 2))
for i in range(REPEAT_TIMES):
    dim_results = np.ndarray(shape=(len(DIMS), 2))
    for DIM_INDEX, NUM_DIMENSIONS in enumerate(DIMS):
        print(NUM_DIMENSIONS)
        # global variables
        NUM_OBSERVATIONS = 20
        BATCH_SIZE = 6
        NUM_ITERS = 5000
        VALIDATION_SPLIT = 0.2
        TEST_SPLIT = 0.1
        VALIDATE_EVERY_K_ITERATIONS = 5
        LEARNING_RATE = 0.001
        LOSS = torch.nn.functional.mse_loss
        USE_WEIGHT = True

        # synthetic-data-only variables
        ADD_NOISE = False


        data_x = None
        data_y = None
        obs_y = None
        expected_y = None
        dep_data_x = None
        dep_data_y = None
        dep_obs_y = None
        meta = None
        valFunc = None

        def val_func_1(x: list[float]) -> np.ndarray:
            # returns array of y values. each y value is a function f(y_i) = (x_1, ..., x_n)
            val = [np.array([dim * dim for dim in x]).sum()]
            return val
        def val_func_2(x: list[float]) -> np.ndarray:
            # returns array of y values. each y value is a function f(y_i) = (x_1, ..., x_n)
            val = np.array([dim % 4 - (dim / 4) for dim in x])
            return val

        val_func = val_func_1

        data_x, data_y, obs_y, meta = generate_data(entry_no=NUM_ENTRIES,
                                                    dim_no=NUM_DIMENSIONS,
                                                    num_observations=NUM_OBSERVATIONS,
                                                    do_add_noise=ADD_NOISE,
                                                    aggregate=aggregate_mean,
                                                    value_func=val_func)
        dep_obs_y, dep_meta = generate_dependent_observations(data_x, data_y, num_observations=NUM_OBSERVATIONS,
                                                              aggregate=aggregate_mean)

        meta_train, meta_validate, meta_test = split_data(
            meta, test_split=TEST_SPLIT, validation_split=VALIDATION_SPLIT, random_state=RANDOM_SEED)
        data_train = Dataset(data_x=data_x, data_y=data_y,
                             obs_y=obs_y, observations=meta_train)
        data_test = Dataset(data_x=data_x, data_y=data_y,
                            obs_y=obs_y, observations=meta_test)
        data_validate = Dataset(
            data_x=data_x, data_y=data_y, obs_y=obs_y, observations=meta_validate)

        dep_meta_train, _, _ = split_data(
            dep_meta, test_split=TEST_SPLIT, validation_split=VALIDATION_SPLIT, random_state=RANDOM_SEED)
        dep_data_train = Dataset(data_x=data_x, data_y=data_y,
                             obs_y=dep_obs_y, observations=dep_meta_train)

        loss_history = []
        independent_model = AggregateModel(classification=False)
        independent_model.get_model_for(data_train)
        dependent_model = AggregateModel(classification=False)
        dependent_model.get_model_for(dep_data_train)

        independent_prediction_history = []
        dependent_prediction_history = []
        independent_quality_history = []
        dependent_quality_history = []

        for iterIndex in trange(NUM_ITERS):
            independent_loss = independent_model.train(dataset=data_train,
                                                   optimizer=optim.Adam(
                                                       independent_model.parameters(), lr=LEARNING_RATE),
                                                   loss=LOSS,
                                                   batch_size=BATCH_SIZE)
            dependent_loss = dependent_model.train(dataset=dep_data_train,
                                                 optimizer=optim.Adam(
                                                     dependent_model.parameters(), lr=LEARNING_RATE),
                                                 loss=LOSS,
                                                 batch_size=BATCH_SIZE)
            loss_history.append(
                {'Obserwacje niezależne': independent_loss, 'Obserwacje zależne': dependent_loss})

            if not iterIndex % VALIDATE_EVERY_K_ITERATIONS:
                with torch.no_grad():
                    data_x_a, independent_predictions = independent_model.test(
                        dataset=data_validate)
                    data_x_s, dependent_predictions = dependent_model.test(
                        dataset=data_validate)
                    # we cant use data validate from dependent observations, because test data is not random
                    independent_prediction_history.append(independent_predictions)
                    dependent_prediction_history.append(dependent_predictions)
                    data_y_qa = torch.tensor(np.array([val_func(x) for x in data_x_a]))
                    data_y_qs = torch.tensor(np.array([val_func(x) for x in data_x_s]))
                    dependent_quality_history.append(torch.nn.functional.mse_loss(data_y_qs.reshape(-1), dependent_predictions.reshape(-1)))
                    independent_quality_history.append(torch.nn.functional.mse_loss(data_y_qa.reshape(-1), independent_predictions.reshape(-1)))

        with torch.no_grad():
            data_x_i, independent_predictions = independent_model.test(
                dataset=data_test)
            data_x_d, dependent_predictions = dependent_model.test(
                dataset=data_test)
        if SHOW_INDIVIDUAL_PLOTS:
            plot_losses(loss_history)
        series = [
            {
                "label": "Obserwacje niezależne",
                "marker": "^",
                "data_x": data_x_i,
                "data_y": independent_predictions
            },
            {
                "label": "Obserwacje zależne",
                "marker": "o",
                "data_x": data_x_d,
                "data_y": dependent_predictions
            },
        ]
        # plot_xy3d(data_x=data_x, expected_y=data_y, series=series, value_func=val_func)
        if SHOW_INDIVIDUAL_PLOTS:
            plot_xy(data_x=data_x, expected_y=data_y,
                    series=series, value_func=val_func, dim=0)
            if NUM_DIMENSIONS > 1:
                plot_xy(data_x=data_x, expected_y=data_y,
                        series=series, value_func=val_func, dim=1)
                plot_xy(data_x=data_x, expected_y=data_y,
                        series=series, value_func=val_func, dim=-1)
        repeat_results[i, DIM_INDEX] = [min(independent_quality_history), min(dependent_quality_history)]
fig, ax = plt.subplots()
plt.plot(DIMS, repeat_results.mean(axis=0)[:, 0], '-r', label="Obserwacje niezależne")
plt.plot(DIMS, repeat_results.mean(axis=0)[:, 1], '-b', label="Obserwacje zależne")
plt.yscale('log')
plt.legend(loc='upper left')
plt.xlabel('Liczba wymiarow')
plt.ylabel('Minimalny błąd średniokwadratowy')
plt.show()
input('enter to continue...')