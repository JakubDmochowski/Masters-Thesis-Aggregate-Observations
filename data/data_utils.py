import torch
from data.dataset import Dataset
from typing import Callable
import numpy as np
from itertools import chain
from sklearn.model_selection import train_test_split
import pandas as pd


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
