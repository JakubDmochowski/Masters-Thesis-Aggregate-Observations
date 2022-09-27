from data.dataset import Observation
import torch
import numpy as np
import os
import pandas as pd
import category_encoders as ce

filepath = os.getcwd() + "/datasets/breast-cancer-2/breast-cancer.data"
CSV_COLUMNS = ["id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
               "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"]


def encodeX(entries: pd.DataFrame) -> torch.tensor:
    return torch.tensor(entries.to_numpy()).float()


def encodeY(entries: pd.DataFrame) -> torch.tensor:
    ce_BE = ce.BinaryEncoder(cols=['diagnosis'])
    entries = ce_BE.fit_transform(entries)
    # encoding = {
    #     'no-recurrence-events': 0,
    #     'recurrence-events': 1
    # }
    # entries.y = entries.y.map(encoding)
    return torch.tensor(entries.to_numpy()).float()


def getRawData() -> list[torch.tensor, torch.tensor]:
    data_x = np.array([])
    contents = pd.read_csv(filepath, header=None)
    contents.columns = CSV_COLUMNS
    data_x = contents[contents.columns[2:-1]]
    data_y = contents[contents.columns[1]]
    return [data_x, data_y]


def retrieveData(num_observations: int) -> list[torch.tensor, torch.tensor, torch.tensor, list[Observation]]:
    data_x, data_y = getRawData()
    data_x = encodeX(data_x)
    data_y = encodeY(data_y)
    obs_y, meta = generateObservations(data_y, num_observations)
    return [data_x, data_y, obs_y, meta]


def generateObservations(data_y: torch.tensor, num_observations: int) -> list[torch.tensor, list[Observation]]:
    # returned data_y is a tensor shaped (entries, values)
    entry_no = len(data_y)
    meta = np.linspace(0, entry_no, entry_no, endpoint=False, dtype=int)
    np.random.shuffle(meta)
    meta = np.array_split(meta, num_observations)
    meta = [Observation(y, i) for i, y in enumerate(meta)]
    obs_y = torch.stack([torch.index_select(
        data_y, 0, torch.tensor(obs.entries_indices)).mean(axis=0) for obs in meta]).float()
    return [obs_y, meta]


def getWeights() -> tuple[float, float]:
    contents = pd.read_csv(filepath, header=None)
    contents.columns = CSV_COLUMNS
    data_y = encodeY(contents[contents.columns[1]])
    BCount = torch.sum(data_y[:, 1])
    ACount = len(contents[contents.columns[1]]) - BCount
    AWeight = ACount / (BCount + ACount)
    BWeight = BCount / (BCount + ACount)
    return AWeight, BWeight
