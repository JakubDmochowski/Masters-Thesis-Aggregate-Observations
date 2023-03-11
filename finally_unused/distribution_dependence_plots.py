from matplotlib import pyplot as plt
import torch
from data.dataset import Observation
import itertools
from data.ctr_normalize import CTRNormalize
from tqdm import tqdm

from typing import Callable
import numpy as np
import re
import networkx as nx
from statistics import mean
import pandas as pd

num_entries = 50000
X_MIN = -4
X_MAX = 4
NUM_DIMS = 2
labels = ["clicks", "sales"]
allfeatures = ["attr_" + str(i) for i in range(0, NUM_DIMS)]
NUM_LABELS = len(labels)

def prob_func(x: list[float]) -> np.ndarray:
    m = max(abs(X_MIN), abs(X_MAX))
    top = abs(np.sum([(m + 1) * (m + 1) for dim in x]))
    p0 = abs(np.sum([(dim + 1) * (dim + 1) for dim in x])) / top
    p = [p0, 1 - p0]
    p /= sum(p)
    return p


def val_func(x: list[float]) -> np.ndarray:
    # returns array of y values. each y value is a function f(y_i) = (x_1, ..., x_n)
    val = float(np.random.choice([0, 1], size=1, p=prob_func(x)))
    return val
    # val = np.array([(dim+1)*(dim+1) for dim in x])
    # return np.mean(val)


def prob_2_func(x: list[float]) -> np.ndarray:
    m = max(abs(X_MIN), abs(X_MAX))
    top = abs(np.sum([(m + 1) * (m + 1) for dim in x]))
    p0 = abs(np.sum([(dim - 1) * (dim - 1) for dim in x])) / top
    p = [p0, 1 - p0]
    p /= sum(p)
    return p


def val_2_func(x: list[float]) -> np.ndarray:
    # returns array of y values. each y value is a function f(y_i) = (x_1, ..., x_n)
    val = float(np.random.choice([0, 1], size=1, p=prob_2_func(x)))
    return val
    # val = np.array([(dim+1)*(dim+1) for dim in x])
    # return np.mean(val)

def aggregate_by(data_y: torch.tensor, meta: list[Observation]):
    def get_entries(indices):
        return torch.index_select(data_y, 0, torch.tensor(indices))

    def aggregate(entries: torch.tensor):
        return entries.mean(axis=0)

    obs_y = torch.stack([aggregate(get_entries(obs.entries_indices)) for obs in meta]).float()
    return obs_y

def generate_independent_observations(data_y: torch.tensor, num_observations: int) -> list[
    torch.tensor, list[Observation]]:
    # returned data_y is a tensor shaped (entries, values)
    entry_no = len(data_y)
    meta = np.linspace(0, entry_no, entry_no, endpoint=False, dtype=int)
    np.random.shuffle(meta)
    meta = np.array_split(meta, num_observations)
    meta = [Observation(x, i) for i, x in enumerate(meta)]

    obs_y = aggregate_by(data_y, meta)
    return [obs_y, meta]

def generate_dependent_observations(data_x: torch.tensor, data_y: torch.tensor, num_observations: int,
                                    dims: list[int]) -> list[torch.tensor, list[Observation]]:
    # returned data_y is a tensor shaped (entries, values)
    meta = np.linspace(X_MIN, X_MAX, abs(X_MAX - X_MIN) * 10 + 1, endpoint=True, dtype=float)

    aggregation_masks = [torch.ones(num_entries, dtype=torch.bool)]
    for dim in dims:
        new_aggregation_masks = []
        for mask in aggregation_masks:
            for prev, curr in zip(meta, meta[1:]):
                dim_mask = torch.logical_and((data_x[:, dim] <= curr), (data_x[:, dim] >= prev))
                aggregation_mask = torch.logical_and(dim_mask, mask)
                new_aggregation_masks.append(aggregation_mask)
            aggregation_masks = new_aggregation_masks
    aggregation_indices = [mask.nonzero(as_tuple=True)[0] for mask in aggregation_masks]
    meta = [indices.numpy().tolist() for indices in aggregation_indices if indices.numel() > 0]
    # meta = [torch.logical_and((data_x[:, dims] <= curr), (data_x[:, dims] >= prev)).nonzero(as_tuple=True)[0] for prev, curr in
    #         zip(meta, meta[1:])]
    # meta = [obs.numpy().tolist() for obs in meta if obs.size(dim=0)]
    meta = [Observation(x, i) for i, x in enumerate(meta)]

    obs_y = aggregate_by(data_y, meta)

    return [obs_y, meta]

def get_data_y(data_x, data_y, dims: list[int], num_observations=10):
    data_x = torch.tensor(data_x)
    data_y = torch.tensor(data_y)
    obs_y, meta = generate_dependent_observations(data_x, data_y, num_observations=num_observations, dims=dims)
    for obs in meta:
        for index in obs.entries_indices:
            data_y[index] = obs_y[obs.value_vec_index]
    return data_y.numpy()

def aggregate_on_features(features, mincount, data):
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
        allpairsdf["sales"] += np.random.normal(0, gaussian_sigma, len(allpairsdf))
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
        allpairsdf["sales"] += np.random.normal(0, gaussian_sigma, len(allpairsdf))
    return allpairsdf

class TestDataGraph(nx.DiGraph):
    def __init__(self):
        super().__init__(self)

    @staticmethod
    def probability(individual_count: int, sum_count: float, mean_count: float, no_objects: int,
                    eps: np.longdouble = np.finfo(np.longdouble).eps):
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

class Path:
    def __init__(self, edges=[]):
        self.edges = edges
        self.nodes = [edges[0][0], *[edge[1] for edge in edges]] if len(edges) else []
        self.attributes = [self.attribute(node) for node in self.nodes]

    @staticmethod
    def attribute(node):
        return re.search(r'^attr_(.*)_val_(.*)$', node).group(1)

    def add_edge(self, edge):
        # if edge[0] is not in self.edges - path is incorrect
        if len(self.edges) == 0:
            new_node = edge[0]
            self.attributes.append(self.attribute(new_node))
            self.nodes.append(new_node)
        new_node = edge[1]
        self.attributes.append(self.attribute(new_node))
        self.nodes.append(new_node)
        self.edges.append(edge)

    def __len__(self):
        return len(self.edges)

class DataGenerator:
    def __init__(self, data_graph: nx.Graph, no_attributes: int, ctr_normalize: Callable,
                 eps: np.float64 = np.finfo(np.float64).eps):
        self.data_graph = data_graph
        self.no_attributes = no_attributes
        self.eps = eps
        self.ctr_normalize = ctr_normalize

    @staticmethod
    def values_from_node(node):
        ret = re.search(r'^attr_(.*)_val_(.*)$', node)
        attr = ret.group(1)
        val = ret.group(2)
        return int(attr), float(val)

    @staticmethod
    def get_probabilities_for(objects, from_objects=None, attr="ct_prob"):
        if from_objects is None:
            from_objects = objects
        probs = [from_objects[entry][attr] for entry in objects]
        probs = [prob / sum(probs) for prob in probs]
        return probs

    def get_random_from(self, objects: list[tuple], from_objects=None):
        probs = self.get_probabilities_for(objects, from_objects)
        try:
            chosen_index = np.random.choice(len([obj for obj in objects]), 1, p=probs)[0]
            # len(objects) != len([obj for obj in objects]), apparently some kind of bug in networkx library
            return list(enumerate(objects))[chosen_index][1], chosen_index
        except ValueError:
            return None

    def get_entry_path(self, initial_node):
        path = Path(edges=[])
        probs = []
        edge, prob = self.get_next_edge(from_node=initial_node, path=path)
        if not edge:
            return path
        path.add_edge(edge)
        probs.append(prob)
        while len(path) != self.no_attributes - 1:
            edge, prob = self.get_next_edge(from_node=edge[1], path=path)
            if not edge:
                return path
            path.add_edge(edge)
            probs.append(prob)
        return path, probs

    def get_next_edge(self, from_node, path):
        def not_in_visited(new_node):
            return re.search(r'^attr_(.*)_val_(.*)$', new_node).group(1) not in path.attributes

        def has_edges_with_other_attributes(new_node):
            for node in path.nodes:
                if not self.data_graph.has_edge(new_node, node):
                    return False
            return True

        def is_viable(edge):
            new_node = edge[1]
            return not_in_visited(new_node) and has_edges_with_other_attributes(new_node)

        viable_neighbor_edges = [edge for edge in self.data_graph.edges(from_node) if is_viable(edge)]
        if len(viable_neighbor_edges) == 0:
            return None

        edge, chosen_index = self.get_random_from(viable_neighbor_edges, self.data_graph.edges())

        clicks = [np.float64(self.data_graph.edges()[edge]["clicks"]) for edge in viable_neighbor_edges]
        counts = [np.float64(self.data_graph.edges()[edge]["count"]) for edge in viable_neighbor_edges]
        click_prob = clicks[chosen_index] / sum(counts)
        count_prob = counts[chosen_index] / sum(counts)
        return edge, [click_prob, count_prob]

    # Average of probabilities for all edges and nodes
    def expected_z_for(self, obj):
        clicks = float(obj["clicks"])
        sales = float(obj["sales"])
        count = float(obj["count"])
        return [self.ctr_normalize(clicks, count, self.eps), self.ctr_normalize(sales, count, self.eps)]

    def expected_z_for_entry(self, path):
        expected_z_aggregates = [*[self.expected_z_for(self.data_graph.edges()[edge]) for edge in path.edges],
                                 *[self.expected_z_for(self.data_graph.nodes()[node]) for node in path.nodes],
                                 *[self.expected_z_for(self.data_graph.nodes()[node]) for node in path.nodes]]
        return np.array(expected_z_aggregates).mean(axis=0)

    def get_entry_data(self, path):
        data_x = [self.values_from_node(node) for node in path.nodes]
        data_x.sort(key=lambda x: x[0])  # sort by attribute id
        data_x = np.array(data_x)[:, 1]
        expected_z = self.expected_z_for_entry(path)
        return data_x, expected_z

    def generate_entry(self):
        nodes = self.data_graph.nodes()
        entry_path = None
        while entry_path is None or len(entry_path) != self.no_attributes - 1:
            initial_node, chosen_index = self.get_random_from(nodes)
            cl_init_prob = (self.get_probabilities_for(nodes, attr="cl_prob"))[chosen_index]
            ct_init_prob = (self.get_probabilities_for(nodes, attr="ct_prob"))[chosen_index]

            entry_path, probs = self.get_entry_path(initial_node)
        expected_z = self.data_graph.global_z_prob * cl_init_prob / ct_init_prob
        # print(cl_init_prob, ct_init_prob)
        for cl_prob, ct_prob in probs:
            expected_z = expected_z * cl_prob / ct_prob
            # print(cl_prob, ct_prob)
        expected_z_1 = np.array([expected_z])
        data_x, expected_z_2 = self.get_entry_data(entry_path)
        return data_x, expected_z_1, expected_z_2


def generate(data_x: np.array, data_y: np.array):
    print("Generating points...")

    data = pd.DataFrame()
    for i, f in enumerate(allfeatures):
        data[f] = np.array(data_x)[:, i]
    for i, l in enumerate(labels):
        data[l] = np.array(data_y).reshape((len(data_y), len(labels)))[:, i]
    gaussian_sigma = 0.3
    aggregates_single = aggregate_on_all_single(allfeatures, data=data, gaussian_sigma=gaussian_sigma)
    aggregates_pairs = aggregate_on_all_pairs(allfeatures, data=data, mincount=0, gaussian_sigma=gaussian_sigma)
    NORMALIZE = True
    FILTER = False
    if FILTER:
        aggregates_pairs = aggregates_pairs[
            (aggregates_pairs.c > 1) & (aggregates_pairs.clicks > 0) & (aggregates_pairs.sales > 0)]
    if NORMALIZE:
        aggregates_pairs = aggregates_pairs.assign(zeros=0)
        aggregates_pairs.c = aggregates_pairs[['clicks', 'sales', 'c', 'zeros']].max(axis=1)
        aggregates_pairs.clicks = aggregates_pairs[['clicks', 'zeros']].max(axis=1)
        aggregates_pairs.sales = aggregates_pairs[['sales', 'zeros']].max(axis=1)
        aggregates_pairs = aggregates_pairs.drop(['zeros'], axis=1)

        aggregates_single = aggregates_single.assign(zeros=0)
        aggregates_single.c = aggregates_single[['clicks', 'sales', 'c', 'zeros']].max(axis=1)
        aggregates_single.clicks = aggregates_single[['clicks', 'zeros']].max(axis=1)
        aggregates_single.sales = aggregates_single[['sales', 'zeros']].max(axis=1)
        aggregates_single = aggregates_single.drop(['zeros'], axis=1)


    data_graph = TestDataGraph()
    data_graph.prep(data_singles=aggregates_single.to_numpy(), data_pairs=aggregates_pairs.to_numpy())
    DG = DataGenerator(data_graph=data_graph, ctr_normalize=CTRNormalize.no_action,
                       no_attributes=len(allfeatures))


    gen_data_x = []
    gen_data_y1 = []
    gen_data_y2 = []
    for i in tqdm(range(num_entries)):
        x, y1, y2 = DG.generate_entry()
        gen_data_x.append([float(entry) for entry in x])
        gen_data_y1.append([float(entry) for entry in y1])
        gen_data_y2.append([float(entry) for entry in y2])

    gen_data_x = np.array(gen_data_x)
    gen_data_y1 = np.array(gen_data_y1)
    gen_data_y2 = np.array(gen_data_y2)
    return gen_data_x, gen_data_y1, gen_data_y2

def plot_compare(data_x, data_y, data_yp, gen_data_x, gen_data_y1, gen_data_y2):
    fig, ax = plt.subplots(2, 2, sharey=True)
    scat1 = ax[0, 0].scatter(x=data_x[:, 0], y=data_x[:, 1], c = data_yp[:, 0], cmap = "gist_rainbow", vmin=0.0, vmax=1.0)
    ax[0, 0].set_title('X, p(Z)')
    scat2 = ax[0, 1].scatter(x=data_x[:, 0], y=data_x[:, 1], c = data_y[:, 0], cmap = "gist_rainbow", vmin=0.0, vmax=1.0)
    ax[0, 1].set_title('X, Z')
    scat3 = ax[1, 0].scatter(x=gen_data_x[:, 0], y=gen_data_x[:, 1],c = gen_data_y1[:, 0], cmap = "gist_rainbow", vmin=0.0, vmax=1.0)
    ax[1, 0].set_title('X, Z\', metoda 1')
    scat4 = ax[1, 1].scatter(x=gen_data_x[:, 0], y=gen_data_x[:, 1],c = gen_data_y2[:, 0], cmap = "gist_rainbow", vmin=0.0, vmax=1.0)
    ax[1, 1].set_title('X, Z\', metoda 2')
    ax[0, 0].set_xlabel('a_1')
    ax[0, 1].set_xlabel('a_1')
    ax[1, 0].set_xlabel('a_1')
    ax[1, 1].set_xlabel('a_1')
    ax[0, 0].set_ylabel('a_2')
    ax[0, 1].set_ylabel('a_2')
    ax[1, 0].set_ylabel('a_2')
    ax[1, 1].set_ylabel('a_2')
    for a in ax.flat:
        a.label_outer()
    fig.colorbar(scat1, ax=ax[0, 0], label='z')
    fig.colorbar(scat2, ax=ax[0, 1], label='z')
    fig.colorbar(scat3, ax=ax[1, 0], label='z')
    fig.colorbar(scat4, ax=ax[1, 1], label='z')
    fig.show()
