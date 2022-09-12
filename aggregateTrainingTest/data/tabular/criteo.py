from data.dataset import Observation
import torch
import numpy as np
import os
import pandas as pd
import category_encoders as ce
from typing import Callable
import sys
import csv
from tqdm import tqdm
import re
import urllib

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

AGG_PAIR_DATA_SOURCE = 'http://go.criteo.net/criteo-privacy-ml-competition-data/aggregated-noisy-data-pairs.csv.gz'
MAIN_DATA_SOURCE = 'https://competitions.codalab.org/my/datasets/download/03b1ddf5-f0a9-48b6-a37c-1fb076143f95'

criteo_dirpath = os.getcwd() + "/datasets/criteo"
prepared_dirpath = criteo_dirpath + "/prepared"
raw_dirpath = criteo_dirpath + "/raw"

aggregated_noisy_singles_filename = "/aggregated_noisy_data_singles.csv"
aggregated_noisy_pairs_filename = "/aggregated_noisy_data_pairs.csv"
small_train_filename = "/small_train.csv"

filepath = prepared_dirpath + small_train_filename
meta_filepath = prepared_dirpath + f"/.meta"
observations_single_source = raw_dirpath + aggregated_noisy_singles_filename
observations_pairs_source = raw_dirpath + aggregated_noisy_pairs_filename

CSV_COLUMNS = ["hash_0", "hash_1", "hash_2", "hash_3", "hash_4", "hash_5", "hash_6", "hash_7",
               "hash_8", "hash_9", "hash_10", "hash_11", "hash_12", "hash_13", "hash_14", "hash_15", "hash_16",
               "hash_17", "hash_18", "click", "sale"]


def downloadAggregatedPairs(raw_dirpath: str, force: bool = False):
    observations_pair_source = raw_dirpath + aggregated_noisy_pairs_filename
    if not os.path.exists(observations_pair_source):
        print("downloading additional data...", end='')
        with urllib.request.urlopen(AGG_PAIR_DATA_SOURCE) as response, open(observations_pair_source, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print('finished:', observations_pair_source)


def downloadCriteo(raw_dirpath: str):
    if os.path.exists(raw_dirpath):
        files = [f for f in os.listdir(
            raw_dirpath) if os.path.isfile(os.path.join(raw_dirpath, f))]
        if 'small_train.csv' in files and 'aggregated_noisy_data_singles.csv' in files:
            return
    else:
        print(
            f"Criteo dataset available at URI: {MAIN_DATA_SOURCE}\nPlease download by hand\nUnzip files and put files [{small_train_filename} and {aggregated_noisy_singles_filename}] to directory \"{raw_dirpath}\"")


def downloadCriteoDataset(raw_dirpath: str) -> None:
    downloadCriteo(raw_dirpath)
    downloadAggregatedPairs(raw_dirpath)


def validateDataset(filename: str = 'observations') -> None:
    small_train = pd.read_csv(filepath)
    observations_meta_destination = prepared_dirpath + f"/{filename}_meta.csv"
    observations_meta_file = open(observations_meta_destination)
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


def prepareCriteoDataset(removeOutliers: bool = False) -> None:
    if not os.path.exists(criteo_dirpath):
        os.makedirs(criteo_dirpath)
    if not os.path.exists(prepared_dirpath):
        os.makedirs(prepared_dirpath)
    if not os.path.exists(raw_dirpath):
        os.makedirs(raw_dirpath)
    downloadCriteoDataset(raw_dirpath)
    small_train_raw_filepath = raw_dirpath + "/small_train.csv"
    small_train_prepared_filepath = prepared_dirpath + "/small_train.csv"
    small_train = pd.read_csv(small_train_raw_filepath)
    meta = getMeta()
    if not 'removeOutliers' in meta:
        meta['removeOutliers'] = removeOutliers
    if removeOutliers is True or 'removeOutliers' in meta and meta['removeOutliers'] != str(removeOutliers):
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
        small_train.drop(index=indices_to_remove, inplace=True)
    small_train.to_csv(small_train_prepared_filepath, index=False)
    setMeta(meta)


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


class CTRNormalize:
    @staticmethod
    def cutoff(clicks: float, count: float, eps: float):
        count = max(count, eps)
        clicks = min(max(clicks, 0), count)
        return clicks / count


def getMeta():
    if not os.path.exists(meta_filepath):
        return {}
    meta_file = open(meta_filepath, "r")
    meta_reader = csv.reader(meta_file)
    meta = {}
    for entry in meta_reader:
        key, value = entry
        meta[key] = value
    return meta


def setMeta(meta):
    meta_file = open(meta_filepath, "w", newline='')
    meta_writer = csv.writer(meta_file)
    for key in meta:
        meta_writer.writerow([key, meta[key]])
    meta_file.close()


def saveMeta(filename):
    meta_file = open(prepared_dirpath + f"/{filename}.meta", "w", newline='')
    meta_writer = csv.writer(meta_file)
    meta = getMeta()
    for key in meta:
        meta_writer.writerow([key, meta[key]])
    meta_file.close()


def _count_generator(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024 * 1024)


def prepareObservations(normalizeCTR: Callable, minCount: float = None, filename: str = 'observations', removeOutliers: bool = False, withPairs: bool = False, force: bool = False) -> None:
    prepareCriteoDataset(removeOutliers)
    observations_destination = prepared_dirpath + f"/{filename}.csv"
    if force or not os.path.exists(observations_destination):
        if os.path.exists(observations_destination):
            os.remove(observations_destination)
        observations_single_source_file = open(observations_single_source)
        with open(observations_single_source, 'rb') as file:
            s_c_generator = _count_generator(file.raw.read)
            single_count = sum(buffer.count(b'\n') for buffer in s_c_generator)
        observations_source_file_single_reader = csv.reader(
            observations_single_source_file, delimiter=',')
        next(observations_source_file_single_reader, None)  # skip the headers
        observations_file = open(observations_destination, "w", newline='')
        observations_file_writer = csv.writer(observations_file, delimiter=';')
        entries = pd.read_csv(filepath)
        removedCountSingle = 0
        for entry in tqdm(observations_source_file_single_reader, total=single_count):
            feature_value, feature_id, count, clicks, sales = entry
            ctr = normalizeCTR(float(clicks), float(count), EPS)
            entries_indices = list(
                np.where(entries[f"hash_{int(feature_id)}"] == int(feature_value))[0])

            if len(entries_indices):
                if minCount is not None and float(count) < minCount:
                    removedCountSingle += 1
                    continue
                observations_file_writer.writerow([entries_indices, ctr])
        print(f"single-observation removed count {removedCountSingle}")
        removedCountPairs = 0
        if withPairs:
            observations_pairs_source_file = open(observations_pairs_source)
            with open(observations_pairs_source, 'rb') as file:
                p_c_generator = _count_generator(file.raw.read)
                pairs_count = sum(buffer.count(b'\n')
                                  for buffer in p_c_generator)
            observations_source_file_pairs_reader = csv.reader(
                observations_pairs_source_file, delimiter=',')
            next(observations_source_file_pairs_reader, None)  # skip the headers
            for entry in tqdm(observations_source_file_pairs_reader, total=pairs_count):
                feature_1_value, feature_2_value, feature_1_id, feature_2_id, count, clicks, sales = entry
                ctr = normalizeCTR(float(clicks), float(count), EPS)
                entries_indices = list(
                    np.where((entries[f"hash_{int(feature_1_id)}"] == int(feature_1_value)) & (entries[f"hash_{int(feature_2_id)}"] == int(feature_2_value)))[0])
                if len(entries_indices):
                    if minCount is not None and float(count) < minCount:
                        removedCountPairs += 1
                        continue
                    observations_file_writer.writerow([entries_indices, ctr])
            print(f"pairs-observation removed count {removedCountPairs}")
            observations_pairs_source_file.close()
        observations_file.close()
        observations_single_source_file.close()
        meta = getMeta()
        meta["withPairs"] = withPairs
        meta["minCount"] = minCount
        setMeta(meta)
    return


def retrieveObservations(filename: str = 'observations') -> list[torch.tensor, list[Observation]]:
    observations_destination = prepared_dirpath + f"/{filename}.csv"
    observations_file = open(observations_destination, 'r')
    observations_file_reader = csv.reader(observations_file, delimiter=';')
    obs_y = []
    meta = []
    value_vec_index = 0
    for entry in observations_file_reader:
        entries_indices, one_prob = entry
        one_prob = float(one_prob)
        obs_y.append([one_prob, 1 - one_prob])
        entries_indices = entries_indices.replace(
            '[', '').replace(']', '').replace(' ', '')
        entries_indices = list(
            map(lambda x: int(x), entries_indices.split(',')))
        observation = Observation(
            entries_indices=entries_indices, value_vec_index=value_vec_index)
        meta.append(observation)
        value_vec_index += 1
    obs_y = torch.tensor(np.array(obs_y))
    observations_file.close()
    return [obs_y, meta]


def getObservations(filename: str = 'observations') -> list[torch.tensor, list[Observation]]:
    prepareObservations(filename)
    return retrieveObservations(filename)


def retrieveData(filename: str = 'observations') -> list[torch.tensor, torch.tensor, torch.tensor, list[Observation]]:
    data_x, data_y = getRawData()
    data_x = encodeX(data_x)
    data_y = encodeY(data_y)
    obs_y, meta = getObservations(filename)
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
