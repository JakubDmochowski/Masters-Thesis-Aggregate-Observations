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

train_file = os.getcwd() + "/datasets/spect/SPECT.train"
test_file = os.getcwd() + "/datasets/spect/SPECT.test"


def encode_data(contents):
    data_x_slc = list(range(contents.shape[1]))
    data_x_slc.remove(0)
    data_x = contents[data_x_slc].to_numpy()
    data_y = contents[0].to_numpy().reshape(-1, 1)
    return data_x, data_y


def get_data_count():
    return len(pd.read_csv(train_file, header=None))

def get_training_data():
    return encode_data(pd.read_csv(train_file, header=None))


def get_testing_data():
    return encode_data(pd.read_csv(test_file, header=None))

def get_weights(normalize = False):
    data_x, data_y = get_training_data()
    b_count = torch.sum(torch.tensor(np.array(data_y)))
    a_count = len(data_x) - b_count
    a_weight = a_count
    b_weight = b_count
    if normalize:
        a_weight /= (b_count + a_count)
        b_weight /= (b_count + a_count)
    return a_weight, b_weight


class SPECTDataGraph(nx.DiGraph):
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
        data = pd.read_csv(train_file, header=None)
        features = data.columns[1:]
        labels = data.columns[0:1]
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
