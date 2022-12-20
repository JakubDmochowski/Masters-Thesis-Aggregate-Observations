import csv
import os
from typing import Callable
import networkx as nx
import numpy as np
import re
from tqdm import tqdm


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
    def __init__(self, data_graph: nx.Graph, no_attributes: int, ctr_normalize: Callable, eps: np.float64 = np.finfo(np.float64).eps):
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
        data_x.sort(key=lambda x: x[0]) # sort by attribute id
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

    def generate_data(self, count: int, filename: str, force: bool = False):
        if os.path.exists(filename):
            if force:
                os.remove(filename)
            else:
                return
        file = open(filename, "w", newline='')
        writer = csv.writer(file)
        attributes = [f"hash_{i}" for i in range(self.no_attributes)]
        headers = [*attributes, "prob_click", "prob_sale"]
        writer.writerow(headers)
        for i in tqdm(range(count), total=count, desc="Generating data"):
            data_x, expected_z = self.generate_entry()
            writer.writerow([*data_x, *expected_z])
        file.close()
