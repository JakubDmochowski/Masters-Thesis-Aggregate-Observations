import torch
import numpy as np
from data.dataset import Observation
from typing import Callable
from data.data_utils import generate_values
import networkx as nx
import tqdm
from statistics import mean

X_MIN = -10
X_MAX = 10


def add_noise(data: torch.tensor) -> torch.tensor:
    return torch.tensor(data.numpy()).float()


def generate_points(entry_no: int, dim_no: int, options: dict) -> torch.tensor:
    # returned data_x is a tensor shaped (entries, features)
    X_MIN = -10
    X_MAX = 10
    if 'x_min' in options:
        X_MIN = options["x_min"]
    if 'x_max' in options:
        X_MAX = options["x_max"]
    data_x = torch.tensor(
        np.random.uniform(X_MIN, X_MAX, entry_no * dim_no
                          ).reshape((entry_no, dim_no))).float()
    return data_x

def aggregate_mean(entries: torch.tensor):
    return entries.mean(axis=0)

def aggregate(entries):
    return aggregate_mean(entries)

def aggregate_by(data_y: torch.tensor, meta: list[Observation], aggregate: Callable):
    def get_entries(indices):
        return torch.index_select(data_y, 0, torch.tensor(indices))


    obs_y = torch.stack([aggregate(get_entries(obs.entries_indices))
                        for obs in meta]).float()
    return obs_y


def generate_independent_observations(data_y: torch.tensor, num_observations: int, aggregate: Callable, do_add_noise: bool = False) -> list[
        torch.tensor, list[Observation]]:
    # returned data_y is a tensor shaped (entries, values)
    entry_no = len(data_y)
    meta = np.linspace(0, entry_no, entry_no, endpoint=False, dtype=int)
    np.random.shuffle(meta)
    meta = np.array_split(meta, num_observations)
    meta = [Observation(x, i) for i, x in enumerate(meta)]
    if do_add_noise is True:
        data_y = add_noise(data_y)

    obs_y = aggregate_by(data_y, meta, aggregate)
    return [obs_y, meta]


def generate_dependent_observations(data_x: torch.tensor, data_y: torch.tensor, num_observations: int, aggregate:Callable,
                                    do_add_noise: bool = False) -> list[
        torch.tensor, list[Observation]]:
    # returned data_y is a tensor shaped (entries, values)
    meta = np.linspace(X_MIN, X_MAX, num_observations,
                       endpoint=False, dtype=float)
    meta = [torch.logical_and((data_x[:, 0] < curr), (data_x[:, 0] >= prev)).nonzero(as_tuple=True)[0] for prev, curr in
            zip(meta, meta[1:])]
    meta = [obs.numpy().tolist() for obs in meta if obs.size(dim=0)]
    meta = [Observation(x, i) for i, x in enumerate(meta)]
    if do_add_noise is True:
        data_y = add_noise(data_y)

    obs_y = aggregate_by(data_y, meta, aggregate)

    return [obs_y, meta]


def generate_data(entry_no: int, num_observations: int, dim_no: int, value_func: Callable, aggregate: Callable, do_add_noise: bool = False,
                  options: dict = {}) -> list[torch.tensor, torch.tensor, torch.tensor, list[Observation]]:
    data_x = generate_points(entry_no, dim_no, options)
    data_y = generate_values(data_x, value_func)
    #     obs_y, meta = generate_dependent_observations(data_x, data_y, num_observations, do_add_noise)
    obs_y, meta = generate_independent_observations(data_y, num_observations, aggregate, do_add_noise)
    return [data_x, data_y, obs_y, meta]


class SyntheticDataGraph(nx.DiGraph):
    def __init__(self):
        super().__init__(self)
        self.no_attributes = None

    @staticmethod
    def probability(individual_count: int, sum_count: float, mean_count: float, no_objects: int, eps: np.longdouble = np.finfo(np.longdouble).eps):
        return (np.longdouble(individual_count) + np.longdouble(mean_count) + eps) / (
                np.longdouble(sum_count) + (np.longdouble(no_objects) * np.longdouble(mean_count)) + eps)

    def get_probabilities_for(self, objects, from_objects=None, attr="count", cattr=None):
        if from_objects is None:
            from_objects = objects
        objects_counts = [float(from_objects[obj][attr]) for obj in objects]
        sum_count = sum(objects_counts)
        if cattr is not None:
            sum_count = sum([float(from_objects[obj][cattr]) for obj in objects])
        mean_count = mean(objects_counts)
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
        for node, prob in tqdm(zip(self.nodes(), nprobs), total=self.number_of_nodes(), desc="Assigning probabilities count"):
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
        for node, prob in tqdm(zip(self.nodes(), nprobs), total=self.number_of_nodes(), desc="Assigning probabilities clicks"):
            node_probabilities[node] = prob
            edge_probs = self.get_probabilities_for(self.edges(node), self.edges(), attr="clicks", cattr="count")
            for edge, eprob in zip(self.edges(node), edge_probs):
                edge_probabilities[edge] = eprob
        nx.set_node_attributes(self, node_probabilities, name="cl_prob")
        del node_probabilities
        nx.set_edge_attributes(self, edge_probabilities, name="cl_prob")
        del edge_probabilities
        del nprobs

    def create_nodes(self, data_singles):
        for entry in data_singles:
            feature_value, clicks, sales, count, feature_id = entry
            node = f"attr_{int(feature_id)}_val_{feature_value}"
            self.add_node(node, count=float(count), clicks=float(clicks), sales=float(sales))

    def create_edges(self, data_pairs):
        for entry in data_pairs:
            feature_1_value, feature_2_value, clicks, sales, count, feature_1_id, feature_2_id = entry
            node_a = f"attr_{int(feature_1_id)}_val_{feature_1_value}"
            node_b = f"attr_{int(feature_2_id)}_val_{feature_2_value}"
            self.add_edge(node_a, node_b, count=float(count), clicks=float(clicks), sales=float(sales))
            self.add_edge(node_b, node_a, count=float(count), clicks=float(clicks), sales=float(sales))

    def prep(self, data_singles, data_pairs):
        self.create_nodes(data_singles)
        self.create_edges(data_pairs)
        self.remove_unreachable_nodes()
        self.assign_probabilities()
        clicks_sum = 0
        count_sum = 0
        for node in self.nodes():
            clicks_sum += np.float64(self.nodes()[node]["clicks"])
            count_sum += np.float64(self.nodes()[node]["count"])
        self.global_z_prob = clicks_sum / count_sum