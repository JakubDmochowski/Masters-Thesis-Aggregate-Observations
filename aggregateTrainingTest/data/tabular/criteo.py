from data.dataset import Observation
import torch
import numpy as np
import os
import pandas as pd
import category_encoders as ce
import sys
import csv
from tqdm import tqdm
import shutil
import errno
import re

# observations meta csv has some fields that are exceeding csv reader's limits
# to overcome this problem we set higher limitation
EPS = 1e-8
maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

prepared_dir_filepath = os.getcwd() + "/datasets/criteo/prepared"
filepath = prepared_dir_filepath + "/small_train.csv"
observations_source = prepared_dir_filepath + \
    "/aggregated_noisy_data_singles.csv"
observations_destination = prepared_dir_filepath + "/observations.csv"
observations_meta_destination = prepared_dir_filepath + "/observations_meta.csv"
CSV_COLUMNS = ["hash_0", "hash_1", "hash_2", "hash_3", "hash_4", "hash_5", "hash_6", "hash_7",
               "hash_8", "hash_9", "hash_10", "hash_11", "hash_12", "hash_13", "hash_14", "hash_15", "hash_16",
               "hash_17", "hash_18", "click", "sale"]


def validateDataset() -> None:
    small_train = pd.read_csv(filepath)
    observations_meta_filepath = prepared_dir_filepath + "/observations_meta.csv"
    observations_meta_file = open(observations_meta_filepath)
    observations_meta = csv.reader(observations_meta_file, delimiter=";")

    occurances = np.zeros(len(small_train))
    for observation in observations_meta:
        entries_indices = [int(x)
                           for x in re.sub(r'\[|\]|\s', '', observation[0])]
        for index in entries_indices:
            occurances[index] += 1

    counts = {}
    for entry in occurances:
        if entry not in counts.keys():
            counts[entry] = 0
        counts[entry] += 1

    print(counts)
    # { 19: 102339, 18: 95 } -> result for raw dataset
    #
    # we want to remove entries, from the dataset,
    # that have 18 occurances instead of 19
    # after preparation we have { 19: 102339 }

    # for index, entry in enumerate(occurances):
    #     if entry == 18:
    #         indices_to_remove.append(index)
    # print(indices_to_remove)


def prepareCriteoDataset(force: bool = False) -> None:
    if force or not os.path.exists(filepath) or not os.path.exists(observations_source):
        if os.path.exists(filepath):
            os.remove(filepath)
        if os.path.exists(observations_source):
            os.remove(observations_source)
        # indices below are the result of the code commented above
        indices_to_remove = [29,
                             1166,
                             3378,
                             4642,
                             4935,
                             5347,
                             6303,
                             6448,
                             7819,
                             9055,
                             9204,
                             12367,
                             14319,
                             14445,
                             15456,
                             16527,
                             17028,
                             17381,
                             17609,
                             17697,
                             19952,
                             20016,
                             20973,
                             21559,
                             21971,
                             22290,
                             24883,
                             25183,
                             25544,
                             26266,
                             26688,
                             27933,
                             31080,
                             37999,
                             38454,
                             39276,
                             41069,
                             42789,
                             45057,
                             46388,
                             47472,
                             47477,
                             47964,
                             48057,
                             48294,
                             51374,
                             53361,
                             54061,
                             54924,
                             60226,
                             60339,
                             61117,
                             61724,
                             62225,
                             63194,
                             64007,
                             64560,
                             64672,
                             64851,
                             65771,
                             66778,
                             67326,
                             67845,
                             67895,
                             68605,
                             70740,
                             70832,
                             71978,
                             73344,
                             74764,
                             76965,
                             78381,
                             78912,
                             79269,
                             79684,
                             81343,
                             84675,
                             85808,
                             86693,
                             87448,
                             88247,
                             90077,
                             91251,
                             94134,
                             94840,
                             96964,
                             97063,
                             99313,
                             99932,
                             100278,
                             100383,
                             100747,
                             100993,
                             101999,
                             102131]
        small_train_filepath = os.getcwd() + "/datasets/criteo/raw/small_train.csv"
        small_train_filepath_dest = os.getcwd(
        ) + "/datasets/criteo/prepared/small_train.csv"
        small_train = pd.read_csv(small_train_filepath)

        small_train.drop(index=indices_to_remove, inplace=True)
        small_train.to_csv(small_train_filepath_dest, index=False)

        aggregated_noisy_data_singles_src = os.getcwd(
        ) + "/datasets/criteo/raw/aggregated_noisy_data_singles.csv"
        aggregated_noisy_data_singles_dest = os.getcwd(
        ) + "/datasets/criteo/prepared/aggregated_noisy_data_singles.csv"

        if not os.path.exists(aggregated_noisy_data_singles_dest):
            if not os.path.exists(aggregated_noisy_data_singles_src):
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), aggregated_noisy_data_singles_src)
            shutil.copyfile(aggregated_noisy_data_singles_src,
                            aggregated_noisy_data_singles_dest)


def encodeX(entries: pd.DataFrame) -> torch.tensor:
    return torch.tensor(entries.to_numpy()).float()


def encodeY(entries: pd.DataFrame) -> torch.tensor:
    ce_BE = ce.BinaryEncoder(cols=['click'])
    entries = ce_BE.fit_transform(entries)
    return torch.tensor(entries.to_numpy()).float()


def getRawData() -> list[torch.tensor, torch.tensor]:
    data_x = np.array([])
    contents = pd.read_csv(filepath)
    contents.columns = CSV_COLUMNS
    data_x = contents[contents.columns[0:-2]]
    data_y = contents[contents.columns[-2:-1]]
    return [data_x, data_y]


def getNormalizedCTR(clicks: float, counts: float, eps: float, type: str = 'cutoff') -> float:
    if type == 'cutoff':
        counts = max(counts, eps)
        clicks = min(max(clicks, 0), counts)
        return clicks / counts
    else:
        availableTypes = ['cutoff', 'mirror', 'resample', 'shift', 'resize']
        raise ValueError(
            f"Criteo getNormalizedCTR - type argument error: passed value {type}, expected to be one of {availableTypes}")


def prepareObservations(force: bool = False, ctr_norm: str = 'cutoff') -> None:
    if not os.path.exists(prepared_dir_filepath):
        os.makedirs(prepared_dir_filepath)
    prepareCriteoDataset(force)
    if force or not os.path.exists(observations_destination) or not os.path.exists(observations_meta_destination):
        if os.path.exists(observations_destination):
            os.remove(observations_destination)
        if os.path.exists(observations_meta_destination):
            os.remove(observations_meta_destination)
        observations_source_file = open(observations_source)
        observations_source_file_reader = csv.reader(
            observations_source_file, delimiter=',')
        next(observations_source_file_reader, None)  # skip the headers
        observations_file = open(observations_destination, "w", newline='')
        observations_file_writer = csv.writer(observations_file, delimiter=';')
        observations_meta_file = open(
            observations_meta_destination, "w", newline='')
        observations_meta_file_writer = csv.writer(
            observations_meta_file, delimiter=';')
        observation_index = 0
        entries = pd.read_csv(filepath)
        for entry in tqdm(observations_source_file_reader):
            feature_value, feature_id, count, clicks, sales = entry
            # ctr may be negative and bigger than 1 -> should normalize
            ctr = getNormalizedCTR(float(clicks), float(count), EPS, ctr_norm)
            entries_indices = list(
                np.where(entries[f"hash_{int(feature_id)}"] == int(feature_value))[0])
            if len(entries_indices):
                observations_file_writer.writerow([ctr, 1-ctr])
                observations_meta_file_writer.writerow(
                    [entries_indices, observation_index])
                observation_index += 1
        observations_file.close()
        observations_meta_file.close()
        observations_source_file.close()
    return


def retrieveObservations() -> list[torch.tensor, list[Observation]]:
    observations_file = open(observations_destination, 'r')
    observations_file_reader = csv.reader(observations_file, delimiter=';')
    obs_y = []
    for entry in observations_file_reader:
        one_prob, no_prob = entry
        obs_y.append([float(one_prob), float(no_prob)])
    obs_y = torch.tensor(np.array(obs_y))
    observations_meta_file = open(observations_meta_destination, 'r')
    observations_meta_file_reader = csv.reader(
        observations_meta_file, delimiter=';')
    meta = []
    for entry in observations_meta_file_reader:
        entries_indices, value_vec_index = entry
        value_vec_index = int(value_vec_index)
        entries_indices = entries_indices.replace(
            '[', '').replace(']', '').replace(' ', '')
        entries_indices = list(
            map(lambda x: int(x), entries_indices.split(',')))
        observation = Observation(
            entries_indices=entries_indices, value_vec_index=value_vec_index)
        meta.append(observation)
    return [obs_y, meta]


def getObservations() -> list[torch.tensor, list[Observation]]:
    prepareObservations()
    return retrieveObservations()


def retrieveData() -> list[torch.tensor, torch.tensor, torch.tensor, list[Observation]]:
    data_x, data_y = getRawData()
    data_x = encodeX(data_x)
    data_y = encodeY(data_y)
    obs_y, meta = getObservations()
    return [data_x, data_y, obs_y, meta]


def getWeights() -> tuple[float, float]:
    prepareCriteoDataset()
    contents = pd.read_csv(filepath)
    contents.columns = CSV_COLUMNS
    data_y = encodeY(contents[contents.columns[19]])
    BCount = torch.sum(data_y[:, 1])
    ACount = len(contents[contents.columns[19]]) - BCount
    AWeight = ACount / (BCount + ACount)
    BWeight = BCount / (BCount + ACount)
    return AWeight, BWeight
