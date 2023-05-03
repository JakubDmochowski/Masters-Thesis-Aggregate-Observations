import csv
from itertools import chain

import lightgbm
import numpy as np
import os

from sklearn.model_selection import train_test_split

from data.data_generator import DataGenerator
from models.gbm.aggregate_model import AggregateModel
from models.gbm.standard_model import StandardModel
from data.dataset import Dataset, Observation
from data.tabular.spect import SPECTDataGraph, get_testing_data, get_training_data, get_weights, get_data_count
from data.data_utils import observation_subset_for, generate_independent_observations
import torch
from torch import optim
from tqdm import trange, tqdm
from data.ctr_normalize import CTRNormalize
from plot_utils import plot_losses, plot_auc, plot_confusion_matrix
from scipy import optimize
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

RESULT_FILENAME = os.getcwd() + '/research_results/spect_gbm_results_2.csv'
if os.path.exists(RESULT_FILENAME):
    print("Result file name already exists")
    exit(1)

with open(RESULT_FILENAME, "w", newline='') as f:
    csvwriter = csv.writer(f, dialect='excel')
    csvwriter.writerow(["iteration", "instances", "model", "accuracy", "auc"])


for iteration in range(5):
    RANDOM_SEED = 2022 + iteration

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    # global variables
    NUM_OBSERVATIONS = 25
    BATCH_SIZE = 32
    NUM_ITERS = 500
    VALIDATE_EVERY_K_ITERATIONS = 5
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    TRAIN_PARAMS = {
        "num_boost_round": NUM_ITERS,
        "early_stopping_rounds": 20
    }

    WEIGHTS = torch.tensor(get_weights(), dtype=torch.float)


    def weighted_nll_loss(predictions, observations):
        return torch.nn.functional.nll_loss(torch.log(predictions + torch.finfo(torch.float64).eps),
                                            torch.argmax(observations, dim=1), weight=WEIGHTS)


    def unweighted_nll_loss(predictions, observations):
        return torch.nn.functional.nll_loss(torch.log(predictions + torch.finfo(torch.float64).eps),
                                            torch.argmax(observations, dim=1))


    LOSS = weighted_nll_loss
    CLASSIFICATION = False

    FORCE_NEW_GENERATION = True
    NUM_INITIAL = get_data_count()

    print(f"Source count: {NUM_INITIAL}")
    for NUM_GENERATED_USED in [1*NUM_INITIAL, 4*NUM_INITIAL, 16*NUM_INITIAL]:
        NUM_GENERATED = NUM_GENERATED_USED
        OBSERVATION_SIZE = 4
        NUM_GEN_OBSERVATIONS = int(NUM_GENERATED_USED / OBSERVATION_SIZE)
        print(f"Generated instance used count: {NUM_GENERATED_USED}")
        print(f"Generated observation count: {NUM_GEN_OBSERVATIONS}")

        generated_data_file_path = os.getcwd() + '\\gen_tabular_spect_generated_big_cliques.csv'

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

        gen_data_z = torch.tensor(np.array(gen_data_z)[0:NUM_GENERATED_USED])
        gen_data_x = torch.tensor(np.array(gen_data_x)[0:NUM_GENERATED_USED])

        data_x, data_z = get_training_data()
        data_x = torch.tensor(data_x, dtype=torch.float32)
        data_z = torch.tensor(data_z, dtype=torch.float32).reshape(-1, 1)


        def aggregate_pow(z: torch.tensor, k):
            return torch.pow(z.mean(axis=0), torch.tensor(k))


        def aggregate(z, k):
            return aggregate_pow(z, k)


        gen_obs_y, gen_meta, k = generate_independent_observations(gen_data_z, NUM_GEN_OBSERVATIONS, NUM_GENERATED_USED, aggregate)
        obs_y, meta, _ = generate_independent_observations(data_z, NUM_OBSERVATIONS, NUM_GENERATED_USED, aggregate, k=k)


        def T(z):
            return aggregate_pow(z, k)


        # for testing/validation purposes only
        test_data_x, test_data_z = get_testing_data()
        test_data_x = torch.tensor(test_data_x, dtype=torch.float32)
        test_data_z = torch.tensor(test_data_z, dtype=torch.float32)
        #
        def batch_data(z, num_observations):
            entry_no = len(z)
            meta = np.linspace(0, entry_no, entry_no, endpoint=False, dtype=int)
            np.random.shuffle(meta)
            meta = np.array_split(meta, num_observations)
            meta = [Observation(x, i) for i, x in enumerate(meta)]
            return meta


        def preprocess_labels(z):
            print("Optimizing T function params for better classification")

            def fitness(k):
                vals = torch.pow(z, torch.tensor(k))
                return abs(np.count_nonzero(vals > 0.5) - (len(z) / 2))  # 40/40 class split distribution in spect dataset

            optimal = optimize.brute(fitness, ranges=[slice(0.01, 2, 0.01)], full_output=True)
            # search for such "k", for which the proportion of "0" to "1" labels is possibly close to initial data
            k = optimal[0][0]
            return torch.round(torch.pow(z, torch.tensor(k)))


        std_gen_meta = batch_data(gen_data_z, NUM_GEN_OBSERVATIONS)
        gen_data_z = preprocess_labels(gen_data_z)
        # std_gen_data_z = preprocess_labels(gen_data_z)


        if len(gen_data_z[0]) == 1:
            gen_obs_y = torch.tensor(np.array(list(map(lambda x: [1 - x[0], x[0]], gen_obs_y)), dtype=np.float64))
            gen_data_z = torch.tensor(np.array(list(map(lambda x: [1 - x[0], x[0]], gen_data_z)), dtype=np.float64))
            test_data_z = torch.tensor(np.array(list(map(lambda x: [1 - x[0], x[0]], test_data_z)), dtype=np.float64))
            # std_gen_data_z = torch.tensor(np.array(list(map(lambda x: [1 - x[0], x[0]], std_gen_data_z)), dtype=np.float64))
        if len(data_z[0]) == 1:
            data_z = torch.tensor(np.array(list(map(lambda x: [1 - x[0], x[0]], data_z)), dtype=np.float64))
            obs_y = torch.tensor(np.array(list(map(lambda x: [1 - x[0], x[0]], obs_y)), dtype=np.float64))

        data_train = Dataset(data_x=gen_data_x, data_y=gen_data_z, obs_y=gen_obs_y, observations=gen_meta)
        std_data_train = Dataset(data_x=data_x, data_y=data_z, obs_y=obs_y, observations=meta)
        std_gen_data_train = Dataset(data_x=gen_data_x, data_y=gen_data_z, obs_y=gen_obs_y, observations=std_gen_meta)
        #
        # gen_meta_train, gen_meta_validate = train_test_split(
        #     gen_meta, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED)
        # data_train = Dataset(data_x=gen_data_x, data_y=gen_data_z,
        #                      obs_y=gen_obs_y, observations=gen_meta_train)
        # data_validate = Dataset(data_x=gen_data_x, data_y=gen_data_z,
        #                      obs_y=gen_obs_y, observations=gen_meta_validate)
        #
        # std_meta_train, std_meta_validate = train_test_split(
        #     meta, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED)
        # std_data_train = Dataset(data_x=data_x, data_y=data_z, obs_y=obs_y, observations=std_meta_train)
        # std_data_validate = Dataset(data_x=data_x, data_y=data_z, obs_y=obs_y, observations=std_meta_validate)
        #
        #
        # std_gen_meta_train, std_gen_meta_validate = train_test_split(
        #     std_gen_meta, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED)
        # std_gen_data_train = Dataset(data_x=gen_data_x, data_y=gen_data_z, obs_y=None, observations=std_gen_meta_train)
        # std_gen_data_validate = Dataset(data_x=gen_data_x, data_y=gen_data_z, obs_y=None, observations=std_gen_meta_validate)

        entry_no = len(test_data_z)
        test_meta = np.linspace(0, entry_no, entry_no, endpoint=False, dtype=int)
        np.random.shuffle(test_meta)
        test_meta = np.array_split(test_meta, NUM_OBSERVATIONS)
        test_meta = [Observation(x, i) for i, x in enumerate(test_meta)]

        data_test = Dataset(data_x=test_data_x, data_y=test_data_z, observations=test_meta)

        aggregate_prediction_history = {}
        standard_prediction_history = {}
        standard_gen_prediction_history = {}


        aggregate_model = AggregateModel(train_params=TRAIN_PARAMS, history=aggregate_prediction_history)
        standard_model = StandardModel(train_params=TRAIN_PARAMS, history=standard_prediction_history)
        standard_gen_model = StandardModel(train_params=TRAIN_PARAMS, history=standard_gen_prediction_history)


        MODEL_TYPE = "gbm"
        # LOAD_MODEL = f"{MODEL_TYPE}_2022-08-09"
        MODEL_DATE = "2022-09-09"
        MODEL_INDEX = "0"
        LOAD = False
        SAVE = False
        LOAD_MODEL = f"{MODEL_DATE}_{MODEL_INDEX}" if LOAD else None

        STANDARD_MODEL_KEY = "std"
        AGGREGATE_MODEL_KEY = "agg"
        STANDARD_GEN_MODEL_KEY = "std_gen"

        if LOAD_MODEL is not None:
            aggregate_model.load(MODEL_TYPE, LOAD_MODEL, AGGREGATE_MODEL_KEY)
            standard_model.load(MODEL_TYPE, LOAD_MODEL, STANDARD_MODEL_KEY)
        else:
            print(f"=============== TRAIN AGGREGATE =================")
            aggregate_model.train(dataset=data_train, test=data_test)
            print(f"=============== TRAIN STANDARD =================")
            standard_model.train(dataset=std_data_train, test=data_test)
            print(f"=============== TRAIN STANDARD GEN =================")
            standard_gen_model.train(dataset=std_gen_data_train, test=data_test)
            if SAVE:
                aggregate_model.save(MODEL_TYPE, AGGREGATE_MODEL_KEY)
                standard_model.save(MODEL_TYPE, STANDARD_MODEL_KEY)
                standard_gen_model.save(MODEL_TYPE, STANDARD_GEN_MODEL_KEY)

        with open(RESULT_FILENAME, "a", newline='') as f:
            csvwriter = csv.writer(f, dialect='excel')
            csvwriter.writerow([f"{iteration}", f"{NUM_GENERATED_USED}", "aggregate", f"{aggregate_model.history['test']['accuracy'][len(aggregate_model.history['test']['accuracy'])-1]}", f"{aggregate_model.history['test']['auc'][len(aggregate_model.history['test']['auc'])-1]}"])
            csvwriter.writerow([f"{iteration}", f"{NUM_GENERATED_USED}", "standard", f"{standard_model.history['test']['accuracy'][len(standard_model.history['test']['accuracy'])-1]}", f"{standard_model.history['test']['auc'][len(standard_model.history['test']['auc'])-1]}"])
            csvwriter.writerow([f"{iteration}", f"{NUM_GENERATED_USED}", "standard on gen", f"{standard_gen_model.history['test']['accuracy'][len(standard_gen_model.history['test']['accuracy'])-1]}", f"{standard_gen_model.history['test']['auc'][len(standard_gen_model.history['test']['auc'])-1]}"])
        # print(f"Accuracy \n proposed: {aggregate_model.history['test']['accuracy'][len(aggregate_model.history['test']['accuracy'])-1]} \n standard: {standard_model.history['test']['accuracy'][len(standard_model.history['test']['accuracy'])-1]} \n standard on generated data: {standard_gen_model.history['test']['accuracy'][len(standard_gen_model.history['test']['accuracy'])-1]}")
        # loss_history = []
        # for i in range(NUM_ITERS):
        #     loss_history.append({
        #         'proposed method': aggregate_model.history['test']['binary_logloss'][min(i, len(aggregate_model.history['test']['binary_logloss']) - 1)],
        #         'standard method': standard_model.history['test']['binary_logloss'][min(i, len(standard_model.history['test']['binary_logloss']) - 1)],
        #         'standard method on gen data': standard_gen_model.history['test']['binary_logloss'][min(i, len(standard_gen_model.history['test']['binary_logloss']) - 1)],
        #     })

        # plot_losses(loss_history)
        # data_x_a, aggregate_predictions = aggregate_model.test(
        #     dataset=data_test)
        # data_x_v_std, standard_predictions = standard_model.test(
        #     dataset=data_test)
        # data_x_v_std_gen, standard_gen_predictions = standard_gen_model.test(
        #     dataset=data_test)
        #
        # aggregate_predictions_binary = [1 if x >= 0.5 else 0 for x in aggregate_predictions]
        # standard_predictions_binary = [1 if x >= 0.5 else 0 for x in standard_predictions]
        # standard_gen_predictions_binary = [1 if x >= 0.5 else 0 for x in standard_gen_predictions]
        #
        # prediction_data = [
        #     {
        #         "label": 'Proposed method',
        #         "auc_history": aggregate_model.history['test']['auc']
        #     },
        #     {
        #         "label": 'Standard method',
        #         "auc_history": standard_model.history['test']['auc']
        #     },
        #     {
        #         "label": 'Standard method on gen. data',
        #         "auc_history": standard_gen_model.history['test']['auc']
        #     },
        # ]
        # plot_auc(models=prediction_data)
        #
        # targets = observation_subset_for(data=test_data_z, dataset=data_test)
        # prediction_data = [
        #     {
        #         "label": 'Proposed method',
        #         "prediction_history": [torch.Tensor([[1-e, e] for e in aggregate_predictions_binary])],
        #     },
        #     {
        #         "label": 'Standard method',
        #         "prediction_history": [torch.Tensor([[1-e, e] for e in standard_predictions_binary])],
        #     },
        #     {
        #         "label": 'Standard method on gen. data',
        #         "prediction_history": [torch.Tensor([[1-e, e] for e in standard_gen_predictions_binary])],
        #     },
        # ]
        # plot_confusion_matrix(prediction_data, targets)
