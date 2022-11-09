import numpy as np
import os

from data.data_generator import DataGenerator
from models.gbm.aggregate_model import AggregateModel
from models.gbm.standard_model import StandardModel
from data.dataset import Dataset
from data.tabular.criteo import retrieve_data, CriteoDataGraph
from data.ctr_normalize import CTRNormalize
from data.data_utils import split_data
import torch
from plot_utils import show_statistics


RANDOM_SEED = 2022

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# global variables
VALIDATION_SPLIT = 0.6
TEST_SPLIT = 0.4
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

data_gen_dest = os.getcwd() + "/datasets/criteo/prepared/generated.csv"

data_graph = CriteoDataGraph()
data_graph.prep()
gen = DataGenerator(data_graph=data_graph, no_attributes=19, ctr_normalize=CTRNormalize.cutoff)
gen.generate_data(size=500000, filename=data_gen_dest)

data_x, data_y, obs_y, meta = retrieve_data(filename=data_gen_dest)

meta_train, meta_validate, meta_test = split_data(
    meta, test_split=TEST_SPLIT, validation_split=VALIDATION_SPLIT, random_state=RANDOM_SEED)
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
    statistics = None
    i = 2
    while(i):
        i -= 1
        data_x, data_y, obs_y, meta = gen.generate_data(observation_size=64, count=32)
        data_train = Dataset(data_x=data_x, data_y=data_y,
                     obs_y=obs_y, observations=meta_train)
        aggregate_model.train(dataset=data_train, validate=data_validate)
        standard_model.train(dataset=data_train, validate=data_validate)
        statistics = show_statistics(data={ "label": "validate", "data": data_validate}, models=[{ "label": "aggregate", "data": aggregate_model}, {"label": "standard", "data": standard_model}], statistics=statistics)
    aggregate_model.save(MODEL_TYPE, AGGREGATE_MODEL_KEY)
    standard_model.save(MODEL_TYPE, STANDARD_MODEL_KEY)

show_statistics(data={ "label": "test", "data": data_test}, models=[{ "label": "aggregate", "data": aggregate_model}, {"label": "standard", "data": standard_model}])
input("Press Enter to continue...")
