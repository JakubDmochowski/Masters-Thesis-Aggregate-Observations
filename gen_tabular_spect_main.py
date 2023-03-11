import csv
from itertools import chain

import numpy as np
import os
from data.data_generator import DataGenerator
from models.XBNet.aggregate_model import AggregateModel
from models.XBNet.standard_model import StandardModel
from data.dataset import Dataset, Observation
from data.tabular.spect import SPECTDataGraph, get_testing_data, get_training_data, get_weights
from data.data_utils import observation_subset_for, generate_independent_observations
import torch
from torch import optim
from tqdm import trange, tqdm
from data.ctr_normalize import CTRNormalize
from plot_utils import plot_losses, plot_auc, plot_confusion_matrix
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

RANDOM_SEED = 2022

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# global variables
NUM_OBSERVATIONS = 10
BATCH_SIZE = 6
NUM_ITERS = 5000
VALIDATE_EVERY_K_ITERATIONS = 5
LEARNING_RATE = 0.001

WEIGHTS = torch.tensor(get_weights(), dtype=torch.float)


def weighted_nll_loss(predictions, observations):
    return torch.nn.functional.nll_loss(torch.log(predictions + torch.finfo(torch.float64).eps),
                                        torch.argmax(observations, dim=1), weight=WEIGHTS)


def unweighted_nll_loss(predictions, observations):
    return torch.nn.functional.nll_loss(torch.log(predictions + torch.finfo(torch.float64).eps),
                                        torch.argmax(observations, dim=1))


LOSS = weighted_nll_loss
CLASSIFICATION = False

FORCE_NEW_GENERATION = False
NUM_GENERATED = 10000
NUM_GEN_OBSERVATIONS = 2500

generated_data_file_path = os.getcwd() + '\\gen_tabular_spect_generated.csv'

if FORCE_NEW_GENERATION or not os.path.exists(generated_data_file_path):
    data_file = open(generated_data_file_path, "w", newline='')
    data_file_writer = csv.writer(data_file, delimiter=';')

    data_graph = SPECTDataGraph()
    data_graph.prep()
    DG = DataGenerator(data_graph=data_graph, ctr_normalize=CTRNormalize.no_action)

    for i in tqdm(range(NUM_GENERATED), desc="Generating new points"):
        x, z = DG.generate_entry()
        x = [float(entry) for entry in x]
        z = [float(entry) for entry in z]
        data_file_writer.writerow([x, z])
    data_file.close()

print("Reading generated data...")
data_file = open(generated_data_file_path, "r")
data_file_reader = csv.reader(data_file, delimiter=';')
gen_data_x = []
gen_data_z = []
for x, z in data_file_reader:
    x = x.replace(
        '[', '').replace(']', '').replace(' ', '')
    x = list(
        map(lambda entry: float(entry), x.split(',')))
    z = z.replace(
        '[', '').replace(']', '').replace(' ', '')
    z = list(
        map(lambda entry: float(entry), z.split(',')))
    gen_data_x.append(x)
    gen_data_z.append(z)
#
# data_x = pd.DataFrame(gen_data_x)
# data_z = pd.DataFrame(gen_data_z)
# data = pd.concat((data_x, data_z), axis=1)
# print(data.head())
# data.hist(bins=15, xlabelsize=8, ylabelsize=8, grid=False)
# plt.tight_layout()
#
#
# f, ax = plt.subplots(figsize=(10, 6))
# corr = data.corr()
# hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
#                  linewidths=.05)
# f.subplots_adjust(top=0.93)
# t = f.suptitle('SPECT Attributes Correlation Heatmap (generated)', fontsize=14)
#
# data_x, data_z = get_training_data()
# data_x = pd.DataFrame(data_x)
# data_z = pd.DataFrame(data_z)
# data = pd.concat((data_x, data_z), axis=1)
# print(data.head())
# data.hist(bins=15, xlabelsize=8, ylabelsize=8, grid=False)
# plt.tight_layout()
#
#
# f, ax = plt.subplots(figsize=(10, 6))
# corr = data.corr()
# hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
#                  linewidths=.05)
# f.subplots_adjust(top=0.93)
# t= f.suptitle('SPECT Attributes Correlation Heatmap (original)', fontsize=14)
# plt.show()
# exit(1)

gen_data_z = torch.tensor(np.array(gen_data_z))
gen_data_x = torch.tensor(np.array(gen_data_x))

data_x, data_z = get_training_data()
data_x = torch.tensor(data_x, dtype=torch.float32)
data_z = torch.tensor(data_z, dtype=torch.float32).reshape(-1, 1)

def aggregate_pow(z: torch.tensor, k):
    return torch.pow(z.mean(axis=0), torch.tensor(k))


def aggregate(z, k):
    return aggregate_pow(z, k)


gen_obs_y, gen_meta, k = generate_independent_observations(gen_data_z, NUM_GEN_OBSERVATIONS, NUM_GENERATED, aggregate)
obs_y, meta, _ = generate_independent_observations(data_z, NUM_OBSERVATIONS, NUM_GENERATED, aggregate, k=k)


def T(z):
    return aggregate_pow(z, k)


# for testing/validation purposes only
test_data_x, test_data_z = get_testing_data()
test_data_x = torch.tensor(test_data_x, dtype=torch.float32)
test_data_z = torch.tensor(test_data_z, dtype=torch.float32)
#

if len(gen_data_z[0]) == 1:
    gen_obs_y = torch.tensor(np.array(list(map(lambda x: [1 - x[0], x[0]], gen_obs_y)), dtype=np.float64))
    gen_data_z = torch.tensor(np.array(list(map(lambda x: [1 - x[0], x[0]], gen_data_z)), dtype=np.float64))
if len(data_z[0]) == 1:
    data_z = torch.tensor(np.array(list(map(lambda x: [1 - x[0], x[0]], data_z)), dtype=np.float64))
    obs_y = torch.tensor(np.array(list(map(lambda x: [1 - x[0], x[0]], obs_y)), dtype=np.float64))

data_train = Dataset(data_x=gen_data_x, data_y=gen_data_z,
                     obs_y=gen_obs_y, observations=gen_meta)
std_data_train = Dataset(data_x=data_x, data_y=data_z, obs_y=obs_y, observations=meta)

entry_no = len(test_data_z)
test_meta = np.linspace(0, entry_no, entry_no, endpoint=False, dtype=int)
np.random.shuffle(test_meta)
test_meta = np.array_split(test_meta, NUM_OBSERVATIONS)
test_meta = [Observation(x, i) for i, x in enumerate(test_meta)]

data_test = Dataset(data_x=test_data_x, data_y=test_data_z, observations=test_meta)

layers_raw = [{'nodes': 32, 'nlin': torch.nn.ReLU(inplace=True), 'norm': False, 'bias': True, 'drop': False},
              {'nodes': 20, 'nlin': torch.nn.ReLU(inplace=True), 'norm': True, 'bias': True, 'drop': False}]

loss_history = []

aggregate_model = AggregateModel(
    classification=CLASSIFICATION, layers_raw=layers_raw)
aggregate_model.get_model_for(data_train)
standard_model = StandardModel(
    classification=False, layers_raw=layers_raw)
standard_model.get_model_for(std_data_train)

aggregate_prediction_history = []
standard_prediction_history = []

# def observation_values(obs_y: torch.tensor, observations):
#     _data_y = np.ndarray(shape=torch.tensor(gen_data_z).shape, dtype=np.float32)
#     for obs in observations:
#         for entry_index in obs.entries_indices:
#             _data_y[entry_index] = obs_y[obs.value_vec_index]
#     return _data_y
#
# gen_data_y_vals = observation_values(gen_obs_y, gen_meta).reshape(-1)
# indices = list(chain(*[obs.entries_indices for obs in meta]))
# data_y_vals = np.array(data_z[indices][:,1])
# gen_data_y_vals.sort()
# data_y_vals.sort()
# gen_indices = np.arange(len(gen_data_y_vals)) / len(gen_data_y_vals)
# indices = np.arange(len(data_y_vals)) / len(data_y_vals)
# fig, ax = plt.subplots()
# ax.plot(gen_indices, gen_data_y_vals, label="gen_data_y")
# ax.plot(indices, data_y_vals, label="data_y")
# ax.set_ylabel('F(t)')
# ax.set_xlabel('t')
# ax.legend()
# plt.show()

for iterIndex in trange(NUM_ITERS):
    aggregate_loss = aggregate_model.train(dataset=data_train,
                                           optimizer=optim.Adam(
                                               aggregate_model.parameters(), lr=LEARNING_RATE),
                                           loss=LOSS,
                                           batch_size=BATCH_SIZE)
    standard_loss = standard_model.train(dataset=std_data_train,
                                           optimizer=optim.Adam(
                                               standard_model.parameters(), lr=LEARNING_RATE),
                                           loss=LOSS,
                                           batch_size=BATCH_SIZE)
    loss_history.append(
        {'proponowana metoda': aggregate_loss, 'uczenie standardowe': standard_loss})

    if not iterIndex % VALIDATE_EVERY_K_ITERATIONS:
        with torch.no_grad():
            data_x_a, aggregate_predictions = aggregate_model.test(
                dataset=data_test)
            aggregate_prediction_history.append(np.argmax(aggregate_predictions, axis=1))
            data_x_v_std, standard_predictions = standard_model.test(
                dataset=data_test)
            standard_prediction_history.append(np.argmax(standard_predictions, axis=1))

plot_losses(loss_history)
targets = observation_subset_for(data=test_data_z, dataset=data_test)
prediction_data = [
    {
        "label": 'proponowana metoda',
        "prediction_history": aggregate_prediction_history,
    },
    {
        "label": 'uczenie standardowe',
        "prediction_history": standard_prediction_history,
    },
]
plot_auc(prediction_data, targets,
         every=VALIDATE_EVERY_K_ITERATIONS)
# plot_precision(prediction_data, targets,
#                every=VALIDATE_EVERY_K_ITERATIONS)
# plot_recall(prediction_data, targets,
#             every=VALIDATE_EVERY_K_ITERATIONS)
# plot_confusion_matrix(prediction_data, targets.reshape(-1, 1))

input("Press Enter to continue...")
