from itertools import chain

from data.dataset import Observation
import torch
import numpy as np
import os
import pandas as pd
import category_encoders as ce
import networkx as nx
from statistics import mean
from tqdm import tqdm

filepath = os.getcwd() + "/datasets/breast-cancer-2/breast-cancer.data"
CSV_COLUMNS = ["id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
               "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
               "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
               "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
               "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
               "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst",
               "fractal_dimension_worst"]
FEATURES = CSV_COLUMNS[2:-1]
LABELS = CSV_COLUMNS[1]


def encode_x(entries: pd.DataFrame) -> torch.tensor:
    return torch.tensor(entries.to_numpy()).float()


def encode_y(entries: pd.DataFrame) -> torch.tensor:
    ce_be = ce.BinaryEncoder(cols=['diagnosis'])
    entries = ce_be.fit_transform(entries)
    return torch.tensor(entries.to_numpy()).float()


def get_raw_data() -> list[torch.tensor, torch.tensor]:
    contents = pd.read_csv(filepath, header=None)
    contents.columns = CSV_COLUMNS
    data_x = contents[contents.columns[2:-1]]
    data_y = contents[contents.columns[1]]
    return [data_x, data_y]


def get_encoded_data() -> list[torch.tensor, torch.tensor]:
    data_x, data_y = get_raw_data()
    return encode_x(data_x), encode_y(data_y)


def retrieve_data(num_observations: int) -> list[torch.tensor, torch.tensor, torch.tensor, list[Observation]]:
    data_x, data_y = get_encoded_data()
    obs_y, meta = generate_observations(data_y, num_observations)
    return [data_x, data_y, obs_y, meta]


def generate_observations(data_y: torch.tensor, num_observations: int) -> list[torch.tensor, list[Observation]]:
    # returned data_y is a tensor shaped (entries, values)
    entry_no = len(data_y)
    meta = np.linspace(0, entry_no, entry_no, endpoint=False, dtype=int)
    np.random.shuffle(meta)
    meta = np.array_split(meta, num_observations)
    meta = [Observation(y, i) for i, y in enumerate(meta)]
    obs_y = torch.stack([torch.index_select(
        data_y, 0, torch.tensor(obs.entries_indices)).mean(axis=0) for obs in meta]).float()
    return [obs_y, meta]


def get_weights() -> tuple[float, float]:
    contents = pd.read_csv(filepath, header=None)
    contents.columns = CSV_COLUMNS
    data_y = encode_y(contents[contents.columns[1]])
    b_count = torch.sum(data_y[:, 1])
    a_count = len(contents[contents.columns[1]]) - b_count
    a_weight = a_count / (b_count + a_count)
    b_weight = b_count / (b_count + a_count)
    return a_weight, b_weight


class BreastCancerDataGraph(nx.DiGraph):
    def __init__(self):
        super().__init__(self)
        self.no_attributes = None

    @staticmethod
    def probability(individual_count: int, sum_count: float, mean_count: float, no_objects: int,
                    eps: np.longdouble = np.finfo(np.longdouble).eps):
        return (np.longdouble(individual_count) + eps) / (
                np.longdouble(sum_count) + eps)
        # return (np.longdouble(individual_count) + np.longdouble(mean_count) + eps) / (
        #         np.longdouble(sum_count) + (np.longdouble(no_objects) * np.longdouble(mean_count)) + eps)

    def get_probabilities_for(self, objects, from_objects=None, attr="count", cattr=None):
        if from_objects is None:
            from_objects = objects
        objects_counts = [float(from_objects[obj][attr]) for obj in objects]
        sum_count = sum(objects_counts)
        mean_count = mean(objects_counts)
        if cattr is not None:
            sum_count = sum([float(from_objects[obj][cattr]) for obj in objects])
            mean_count = mean([float(from_objects[obj][cattr]) for obj in objects])
        return np.array(
            [self.probability(individual_count, sum_count, mean_count, len(objects_counts)) for individual_count in
             objects_counts])

    def remove_unreachable_nodes(self):
        to_delete = []
        for node in tqdm(self.nodes(), total=self.number_of_nodes(), desc="Removing unreachable nodes"):
            node_edges = self.edges(node)
            if len(node_edges) == 0:
                to_delete.append(node)
        for node in to_delete:
            self.remove_node(node)
        del to_delete

    def assign_probabilities(self):
        node_probabilities = {}
        edge_probabilities = {}
        nprobs = self.get_probabilities_for(self.nodes(), attr="count")
        for node, prob in tqdm(zip(self.nodes(), nprobs), total=self.number_of_nodes(),
                               desc="Assigning probabilities count"):
            node_probabilities[node] = prob
            edge_probs = self.get_probabilities_for(self.edges(node), self.edges(), attr="count")
            for edge, eprob in zip(self.edges(node), edge_probs):
                edge_probabilities[edge] = eprob
        nx.set_node_attributes(self, node_probabilities, name="ct_prob")
        del node_probabilities
        nx.set_edge_attributes(self, edge_probabilities, name="ct_prob")
        del edge_probabilities
        del nprobs
        node_probabilities = {}
        edge_probabilities = {}
        nprobs = self.get_probabilities_for(self.nodes(), attr="clicks", cattr="count")
        for node, prob in tqdm(zip(self.nodes(), nprobs), total=self.number_of_nodes(),
                               desc="Assigning probabilities clicks"):
            node_probabilities[node] = prob
            edge_probs = self.get_probabilities_for(self.edges(node), self.edges(), attr="clicks", cattr="count")
            for edge, eprob in zip(self.edges(node), edge_probs):
                edge_probabilities[edge] = eprob
        nx.set_node_attributes(self, node_probabilities, name="cl_prob")
        del node_probabilities
        nx.set_edge_attributes(self, edge_probabilities, name="cl_prob")
        del edge_probabilities
        del nprobs

    def aggregate_on_features(self, features, labels, mincount, data):
        df = data[chain(labels, features)]
        df["c"] = 1
        df = df.groupby(features).sum().reset_index()
        df = df[df.c > mincount].copy()
        return df

    def aggregate_on_all_single(self,
                                allfeatures, labels, data, mincount=0
                                ):
        allpairsdf = pd.DataFrame()
        for f0 in allfeatures:
            print("aggregating on", f0)

            features = [f0]
            df = self.aggregate_on_features(features, labels, mincount, data)
            df["feature_1_id"] = int(f0)
            df = df.rename({features[0]: "feature_1_value"}, axis=1)
            allpairsdf = pd.concat([allpairsdf, df])
        return allpairsdf

    def get_data(self):
        contents = pd.read_csv(filepath, header=None)
        features = contents.columns[2:-1]
        labels = contents.columns[1:2]
        data_x, data_y = get_encoded_data()
        data = pd.DataFrame()
        for i, f in enumerate(features):
            data[f] = np.array(data_x)[:, i]
        data[1] = np.array(data_y)[:, 1]
        return data, features, labels

    def aggregate_on_all_pairs(
            self,
            allfeatures,
            labels,
            data,
            mincount=0
    ):
        allpairsdf = pd.DataFrame()
        for f0 in allfeatures:
            feature_1_id = int(f0)
            for f1 in allfeatures:
                feature_2_id = int(f1)
                if not feature_1_id < feature_2_id:
                    continue
                print("aggregating on", f0, f1)
                features = [f0, f1]
                df = self.aggregate_on_features(features, labels, mincount, data)
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
        return allpairsdf

    def create_nodes(self):
        data, features, labels = self.get_data()
        self.no_attributes = len(features)
        data_singles = self.aggregate_on_all_single(features, labels, data=data).to_numpy()
        for entry in data_singles:
            feature_value, clicks, count, feature_id = entry
            node = f"attr_{int(feature_id)}_val_{feature_value}"
            self.add_node(node, count=float(count), clicks=float(clicks))

    def create_edges(self):
        data, features, labels = self.get_data()
        data_pairs = self.aggregate_on_all_pairs(features, labels, data=data).to_numpy()
        for entry in data_pairs:
            feature_1_value, feature_2_value, clicks, count, feature_1_id, feature_2_id = entry
            node_a = f"attr_{int(feature_1_id)}_val_{feature_1_value}"
            node_b = f"attr_{int(feature_2_id)}_val_{feature_2_value}"
            self.add_edge(node_a, node_b, count=float(count), clicks=float(clicks))
            self.add_edge(node_b, node_a, count=float(count), clicks=float(clicks))

    def prep(self):
        self.create_nodes()
        self.create_edges()
        self.remove_unreachable_nodes()
        self.assign_probabilities()
        self.clicks_sum = 0
        self.count_sum = 0
        for node in self.nodes():
            self.clicks_sum += np.float64(self.nodes()[node]["clicks"])
            self.count_sum += np.float64(self.nodes()[node]["count"])
        self.global_z_prob = self.clicks_sum / self.count_sum
