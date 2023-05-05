import torch
from data.dataset import Dataset, Observation
from typing import Callable, Union, Iterable, Tuple
import numpy as np
from itertools import chain
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy import optimize


def observation_subset_for(data: torch.tensor, dataset: Dataset):
    data_indices = list(
        chain(*[obs.entries_indices for obs in dataset.observations]))
    return data[data_indices]


def generate_values(data_x: torch.tensor, value_func: Callable) -> np.ndarray:
    # returned data_y is a tensor shaped (entries, values)
    return torch.tensor(np.array([value_func(x) for x in data_x.numpy()])).float()


def split_data(meta, test_split, validation_split, random_state):
    vt_size = validation_split + test_split
    meta_train, meta_other = train_test_split(
        meta, test_size=vt_size, random_state=random_state)

    meta_validation, meta_test = train_test_split(
        meta_other, test_size=validation_split / vt_size, random_state=random_state)
    return meta_train, meta_validation, meta_test


def get_observations(data_y: torch.tensor, meta: list[Observation], aggregate: Callable, threshold: float):
    def get_entries(indices):
        return torch.index_select(data_y, 0, torch.tensor(indices))

    obs_y = torch.stack([aggregate(get_entries(obs.entries_indices), threshold) for obs in meta]).float()
    return obs_y


def observation_values(data_z, obs_y: torch.tensor, observations):
    _data_z = np.ndarray(shape=data_z.shape, dtype=np.float32)
    for obs in observations:
        for entry_index in obs.entries_indices:
            _data_z[entry_index] = obs_y[obs.value_vec_index]
    return _data_z


def generate_independent_observations(data_z: torch.tensor, num_observations: int, num_generated: int,
                                      aggregate: Callable, k: float = None, k_search_range: Union[Iterable, Tuple] = None) -> list[torch.tensor, list[Observation]]:
    if num_observations >= num_generated / 2:
        raise "Too big number of observations. Each observation must consist of minimum 2 points."
    # returned data_z is a tensor shaped (entries, values)
    entry_no = len(data_z)
    meta = np.linspace(0, entry_no, entry_no, endpoint=False, dtype=int)
    np.random.shuffle(meta)
    meta = np.array_split(meta, num_observations)
    meta = [Observation(x, i) for i, x in enumerate(meta)]
    if k is None:
        if k_search_range is None:
            raise "Exponent k search range not provided."
        print("Optimizing T function params for better classification")
        def fitness(k):
            obs_y = get_observations(data_z, meta, aggregate, k)
            vals = observation_values(data_z, obs_y, meta)
            return abs(np.count_nonzero(vals > 0.5) - (num_generated / 2))

        optimal = optimize.brute(fitness, ranges=k_search_range, full_output=True)
        # search for such "k", for which the proportion of "0" to "1" labels is possibly close to initial data
        k = optimal[0][0]
    obs_y = get_observations(data_z, meta, aggregate, k)
    return obs_y, meta, k


def aggregate_on_features(labels, features, mincount, data):
    df = data[labels + features]
    df["c"] = 1
    df = df.groupby(features).sum().reset_index()
    df = df[df.c > mincount].copy()
    return df


def aggregate_on_all_pairs(
        allfeatures,
        data,
        mincount=0,
        gaussian_sigma=None,
):
    allpairsdf = pd.DataFrame()
    for f0 in allfeatures:
        feature_1_id = int(f0.split("_")[-1])
        for f1 in allfeatures:
            feature_2_id = int(f1.split("_")[-1])
            if not feature_1_id < feature_2_id:
                continue
            print("aggregating on", f0, f1)
            features = [f0, f1]
            df = aggregate_on_features(features, mincount, data)
            df["feature_1_id"] = feature_1_id
            df["feature_2_id"] = feature_2_id
            df = df.rename(
                {
                    features[0]: "feature_1_value",
                    features[1]: "feature_2_value",
                },
                axis=1,
            )
            allpairsdf = pd.concat([allpairsdf, df])
    if gaussian_sigma is not None:
        allpairsdf["c"] += np.random.normal(0, gaussian_sigma, len(allpairsdf))
        allpairsdf["clicks"] += np.random.normal(0, gaussian_sigma, len(allpairsdf))
    return allpairsdf


def aggregate_on_all_single(
        allfeatures, data, mincount=0, gaussian_sigma=None
):
    allpairsdf = pd.DataFrame()
    for f0 in allfeatures:
        print("aggregating on", f0)

        features = [f0]
        df = aggregate_on_features(features, mincount, data)
        df["feature_1_id"] = int(f0.split("_")[-1])
        df = df.rename({features[0]: "feature_1_value"}, axis=1)
        allpairsdf = pd.concat([allpairsdf, df])
    if gaussian_sigma is not None:
        allpairsdf["c"] += np.random.normal(0, gaussian_sigma, len(allpairsdf))
        allpairsdf["clicks"] += np.random.normal(0, gaussian_sigma, len(allpairsdf))
    return allpairsdf
