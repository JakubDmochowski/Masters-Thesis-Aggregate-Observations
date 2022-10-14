import csv
import os
from typing import Callable
import networkx as nx
import numpy as np
import re
from statistics import mean
from tqdm import tqdm


class DataGenerator:
    def __init__(self, data_graph: nx.Graph, no_attributes: int, ctr_normalize: Callable, eps: float = 1e-8):
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

    def get_probabilities_for(self, objects, from_objects=None):
        if from_objects is None:
            from_objects = objects
        probs = [from_objects[entry]["prob"] for entry in objects]
        probs /= sum(probs)
        return probs

    def get_random_from(self, objects: list[tuple], from_objects=None):
        probs = self.get_probabilities_for(objects, from_objects)
        try:
            chosen_index = np.random.choice(len([obj for obj in objects]), 1, p=probs)[0]
            # len(objects) != len([obj for obj in objects]), apparently some kind of bug in networkx library
            return list(enumerate(objects))[chosen_index][1]
        except ValueError:
            return None

    def next(self, data_x, node):
        # print(f"{node} neighbors_count: {len(list(self.data_graph.neighbors(node)))}")
        visited_attributes = [str(key) for key, value in data_x]

        # def filter_edges(visited_attributes):
        #     return lambda a, b: re.search(r'^attr_(.*)_val_(.*)$', b).group(1) not in visited_attributes
        #
        # view = nx.subgraph_view(self.data_graph, filter_edge=filter_edges(visited_attributes))
        # viable_neighbor_edges = nx.edge_subgraph(view, nx.edges(view, node)).edges()
        def not_in_visited(edge):
            return re.search(r'^attr_(.*)_val_(.*)$', edge[1]).group(1) not in visited_attributes

        def has_edges_with_other_attributes(edge):
            new_node = edge[1]
            return self.data_graph.has_edge()

        viable_neighbor_edges = [edge for edge in self.data_graph.edges(node) if not_in_visited(edge) and has_edges_with_other_attributes(edge)]
        if len([edge for edge in viable_neighbor_edges]) == 0:
            return None
        ret_edge = self.get_random_from(viable_neighbor_edges, self.data_graph.edges())
        if ret_edge is None:
            return None
        return ret_edge

    def get_entry_data(self, data_x: list[tuple[int, float]], expected_z_from_aggregates: list[tuple[str, float]], node):
        if len(data_x) == self.no_attributes:
            # print("get_entry_data - enough")
            return data_x, expected_z_from_aggregates
        step = self.next(data_x, node)
        if step is None:
            return data_x, expected_z_from_aggregates
        else:
            new_edge = step
            new_node = new_edge[1]
        data_x.append(self.values_from_node(new_node))
        if self.data_graph.has_edge(*new_edge):
            expected_z_from_aggregates.append((f"edge:{new_edge}", self.expected_z_from(self.data_graph.edges(), new_edge)))
        expected_z_from_aggregates.append((f"node:{new_node}", self.expected_z_from(self.data_graph.nodes(), new_node)))
        return self.get_entry_data(data_x, expected_z_from_aggregates, new_node)


    def expected_z_from(self, objects, obj):
        clicks = float(objects[obj]["clicks"])
        sales = float(objects[obj]["sales"])
        count = float(objects[obj]["count"])
        return [self.ctr_normalize(clicks, count, self.eps), self.ctr_normalize(sales, count, self.eps)]

    @staticmethod
    def expected_z_for_entry(expected_z_from_aggregates: list[tuple[str, list[float]]]):
        return mean([val[0] for val_from, val in expected_z_from_aggregates]), mean([val[1] for val_from, val in expected_z_from_aggregates])

    def generate_entry(self):
        nodes = self.data_graph.nodes()
        data_x = []
        while data_x is None or len(data_x) != self.no_attributes:
            initial_node = self.get_random_from(nodes)
            # print(f"initial node: {initial_node}")
            initial_data = [self.values_from_node(initial_node)]
            data_x, expected_z_from_aggegates = self.get_entry_data(initial_data, [], node=initial_node)
        data_x.sort(key=lambda x: x[0]) # sort by attribute id
        expected_z = np.array(self.expected_z_for_entry(expected_z_from_aggegates))
        data_x = np.array(data_x)[:, 1]
        return data_x, expected_z

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
            entry = self.generate_entry()
            data_x, expected_z = entry
            writer.writerow([*data_x, *expected_z])
        file.close()
