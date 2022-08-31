from data.dataset import Observation
import torch
import numpy as np
import os
import pandas as pd
import category_encoders as ce

filepath = os.getcwd() + "/datasets/breast-cancer-1/breast-cancer.data"


def encodeX(entries: pd.DataFrame) -> torch.tensor:
    ce_OHE = ce.OneHotEncoder(
        cols=['menopause', 'breast', 'breast-quad'])
    entries = ce_OHE.fit_transform(entries)
    ce_BE = ce.BinaryEncoder(cols=['node-caps', 'irradiat'])
    entries = ce_BE.fit_transform(entries)
    tumorSizeEncoding = {
        '0-4': 0,
        '5-9': 5,
        '10-14': 10,
        '15-19': 15,
        '20-24': 20,
        '25-29': 25,
        '30-34': 30,
        '35-39': 35,
        '40-44': 40,
        '45-49': 45,
        '50-54': 50,
        '55-59': 55,
    }
    entries["tumor-size"] = entries["tumor-size"].map(tumorSizeEncoding)
    invNodesEncoding = {
        '0-2': 0,
        '3-5': 3,
        '6-8': 6,
        '9-11': 9,
        '12-14': 12,
        '15-17': 15,
        '18-20': 18,
        '21-23': 21,
        '24-26': 24,
        '27-29': 27,
        '30-32': 30,
        '33-35': 33,
        '36-39': 36,
    }
    entries["inv-nodes"] = entries["inv-nodes"].map(invNodesEncoding)
    ageEncoding = {
        '20-29': 20,
        '30-39': 30,
        '40-49': 40,
        '50-59': 50,
        '60-69': 60,
        '70-79': 70,
    }
    entries.age = entries.age.map(ageEncoding)
    return torch.tensor(entries.to_numpy()).float()


def encodeY(entries: pd.DataFrame) -> torch.tensor:
    ce_BE = ce.BinaryEncoder(cols=['y'])
    entries = ce_BE.fit_transform(entries)
    # encoding = {
    #     'no-recurrence-events': 0,
    #     'recurrence-events': 1
    # }
    # entries.y = entries.y.map(encoding)
    return torch.tensor(entries.to_numpy()).float()


def getRawData() -> list[torch.tensor, torch.tensor]:
    data_x = np.array([])
    contents = pd.read_csv(filepath)
    contents.columns = ['y', 'age', 'menopause',
                        'tumor-size', 'inv-nodes',
                        'node-caps', 'deg-malig',
                        'breast', 'breast-quad',
                        'irradiat']
    data_x = contents.filter(regex='[^y]', axis=1)
    data_y = contents.filter(regex='y', axis=1)
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
    _, data_y = getRawData()
    BCount = torch.sum(encodeY(data_y)[:, 1])
    ACount = len(data_y) - BCount
    AWeight = ACount / (BCount + ACount)
    BWeight = BCount / (BCount + ACount)
    return AWeight, BWeight
