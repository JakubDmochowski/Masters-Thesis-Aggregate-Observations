import math
from typing import Callable
import networkx as nx
import numpy as np
import re
from statistics import mean


class DataGenerator:
    def __init__(self, data_graph: nx.Graph, no_attributes: int, ctr_normalize: Callable, eps: float = 1e-8):
        self.data_graph = data_graph
        edges = data_graph.edges()
        self.no_attributes = no_attributes
        self.eps = eps
        self.ctr_normalize = ctr_normalize

    @staticmethod
    def values_from_node(node):
        ret = re.search(r'^attr_(.*)_val_(.*)$', node)
        attr = ret.group(1)
        val = ret.group(2)
        return int(attr), float(val)

    def get_probabilities_for(self, objects):
        probs = [objects[entry]["prob"] for entry in objects]
        probs /= sum(probs)
        return probs

    def get_random_from(self, objects: list[tuple]):
        probs = self.get_probabilities_for(objects)
        try:
            chosen_index = np.random.choice(len([obj for obj in objects]), 1, p=probs)[0]
            # len(objects) != len([obj for obj in objects]), apparently some kind of bug in networkx library
            return list(enumerate(objects))[chosen_index][1]
        except ValueError:
            return None

    def next(self, data_x, node):
        # print(f"{node} neighbors_count: {len(list(self.data_graph.neighbors(node)))}")
        visited_attributes = [str(key) for key, value in data_x]

        def filter_edges(visited_attributes):
            return lambda a, b: re.search(r'^attr_(.*)_val_(.*)$', b).group(1) not in visited_attributes

        view = nx.subgraph_view(self.data_graph, filter_edge=filter_edges(visited_attributes))
        viable_neighbor_edges = nx.edge_subgraph(view, nx.edges(view, node)).edges()
        if len([edge for edge in viable_neighbor_edges]) == 0:
            return None
        ret_edge = self.get_random_from(viable_neighbor_edges)
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
        count = float(objects[obj]["count"])
        return self.ctr_normalize(clicks, count, self.eps)

    @staticmethod
    def expected_z_for_entry(expected_z_from_aggregates: list[tuple[str, float]]):
        return mean([val for val_from, val in expected_z_from_aggregates])

    def generate_entry(self):
        nodes = self.data_graph.nodes()
        data_x = []
        while data_x is None or len(data_x) != self.no_attributes:
            initial_node = self.get_random_from(nodes)
            # print(f"initial node: {initial_node}")
            initial_data = [self.values_from_node(initial_node)]
            data_x, expected_z_from_aggegates = self.get_entry_data(initial_data, [], node=initial_node)
        data_x.sort(key=lambda x: x[0]) # sort by attribute id
        expected_z = self.expected_z_for_entry(expected_z_from_aggegates)
        data_x = np.array(data_x)[:, 1]
        return data_x, expected_z

    # def generate_entries(self, count: int):
    #     initial_nodes = choice(self.data_graph.edges(), count, p=self.probs)
    #     for node in initial_nodes:
    #

    def generate_data(self, size: int):

