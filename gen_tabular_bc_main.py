import csv
import numpy as np
from sklearn.model_selection import train_test_split
import os
from data.data_generator import DataGenerator
from models.XBNet.aggregate_model import AggregateModel
from data.dataset import Dataset, Observation
from data.tabular.breast_cancer_2 import get_weights, BreastCancerDataGraph, get_encoded_data
from data.data_utils import observation_subset_for, generate_independent_observations
import torch
from torch import optim
from tqdm import trange, tqdm
from data.ctr_normalize import CTRNormalize
from plot_utils import plot_losses, plot_auc, plot_precision, plot_recall, plot_confusion_matrix

RANDOM_SEED = 2022

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# global variables
NUM_OBSERVATIONS = 100
BATCH_SIZE = 6
NUM_ITERS = 5000
VALIDATE_EVERY_K_ITERATIONS = 5
LEARNING_RATE = 0.0005

WEIGHTS = torch.tensor(get_weights(normalize=False), dtype=torch.float)


def weighted_nll_loss(predictions, observations):
    return torch.nn.functional.nll_loss(torch.log(predictions + torch.finfo(torch.float64).eps),
                                        torch.argmax(observations, dim=1), weight=WEIGHTS)


def unweighted_nll_loss(predictions, observations):
    return torch.nn.functional.nll_loss(torch.log(predictions + torch.finfo(torch.float64).eps),
                                        torch.argmax(observations, dim=1))


LOSS = weighted_nll_loss
CLASSIFICATION = False

FORCE_NEW_GENERATION = False
NUM_GENERATED = 1000
NUM_GEN_OBSERVATIONS = 100

generated_data_file_path = os.getcwd() + '\\gen_tabular_generated.csv'

if FORCE_NEW_GENERATION or not os.path.exists(generated_data_file_path):
    data_file = open(generated_data_file_path, "w", newline='')
    data_file_writer = csv.writer(data_file, delimiter=';')

    data_graph = BreastCancerDataGraph()
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
gen_data_z = torch.tensor(np.array(gen_data_z))
gen_data_x = torch.tensor(np.array(gen_data_x))


def aggregate_pow(z: torch.tensor, k):
    return torch.pow(z.mean(axis=0), torch.tensor(k))


def aggregate(z, k):
    return aggregate_pow(z, k)


gen_obs_y, gen_meta, k = generate_independent_observations(gen_data_z, NUM_GEN_OBSERVATIONS, NUM_GENERATED, aggregate)


def T(z):
    return aggregate_pow(z, k)


# for testing/validation purposes only
data_x, data_z = get_encoded_data()
#

if len(gen_data_z[0]) == 1:
    gen_obs_y = torch.tensor(np.array(list(map(lambda x: [1 - x[0], x[0]], gen_obs_y)), dtype=np.float64))
    gen_data_z = torch.tensor(np.array(list(map(lambda x: [1 - x[0], x[0]], gen_data_z)), dtype=np.float64))

data_train = Dataset(data_x=gen_data_x, data_y=gen_data_z,
                     obs_y=gen_obs_y, observations=gen_meta)

entry_no = len(data_z)
meta = np.linspace(0, entry_no, entry_no, endpoint=False, dtype=int)
np.random.shuffle(meta)
meta = np.array_split(meta, NUM_OBSERVATIONS)
meta = [Observation(x, i) for i, x in enumerate(meta)]

meta_test, meta_validate = train_test_split(meta, test_size=0.5, random_state=RANDOM_SEED)
data_test = Dataset(data_x=data_x, data_y=data_z, observations=meta_test)
data_validate = Dataset(
    data_x=data_x, data_y=data_z, observations=meta_validate)

layers_raw = [{'nodes': 32, 'nlin': torch.nn.ReLU(inplace=True), 'norm': False, 'bias': True, 'drop': False},
              {'nodes': 20, 'nlin': torch.nn.ReLU(inplace=True), 'norm': True, 'bias': True, 'drop': False}]

loss_history = []
aggregate_model = AggregateModel(
    classification=CLASSIFICATION, layers_raw=layers_raw)
aggregate_model.get_model_for(data_train)

aggregate_prediction_history = []

for iterIndex in trange(NUM_ITERS):
    aggregate_loss = aggregate_model.train(dataset=data_train,
                                           optimizer=optim.Adam(
                                               aggregate_model.parameters(), lr=LEARNING_RATE),
                                           loss=LOSS,
                                           batch_size=BATCH_SIZE)
    loss_history.append(
        {'aggregate': aggregate_loss})

    if not iterIndex % VALIDATE_EVERY_K_ITERATIONS:
        with torch.no_grad():
            data_x_a, aggregate_predictions = aggregate_model.test(
                dataset=data_validate)
            aggregate_prediction_history.append(aggregate_predictions)

with torch.no_grad():
    data_x_a, aggregate_predictions = aggregate_model.test(
        dataset=data_test)

plot_losses(loss_history)
targets = observation_subset_for(data=data_z, dataset=data_validate)
prediction_data = [
    {
        "label": 'uczenie Zhanga',
        "prediction_history": aggregate_prediction_history,
    },
]
test_prediction_data = [
    {
        "label": 'uczenie Zhanga',
        "prediction_history": [aggregate_predictions],
    },
]
test_targets = observation_subset_for(data=data_z, dataset=data_test)
plot_auc(prediction_data, targets,
         every=VALIDATE_EVERY_K_ITERATIONS)
plot_precision(prediction_data, targets,
               every=VALIDATE_EVERY_K_ITERATIONS)
plot_recall(prediction_data, targets,
            every=VALIDATE_EVERY_K_ITERATIONS)
plot_confusion_matrix(prediction_data, targets)
plot_confusion_matrix(test_prediction_data, test_targets)

input("Press Enter to continue...")
