from algorithms.louvain_core import LouvainCoreAlgorithm
import numpy as np
import networkx as nx
from collections import Counter, defaultdict


class HierarchicalLabelPropagation(LouvainCoreAlgorithm):
    def initialize(self, G):
        initial_partition_map = dict(enumerate(G.nodes()))
        self.levels = []
        self.stats = {"local_moving": []}
        self.levels.append(initial_partition_map)
        # initial_fitness = self.fitness_function(initial_partition_map, G)
        # self.null_fitness.append(initial_fitness)
        self.level_fitness.append(0)
        self.level_graphs.append(G)
        self.gain_stats = []
        return G, initial_partition_map

    def local_movement(self, G, partition_map):

        num_changes = 0
        partitions = np.unique(list(partition_map.values()))
        num_partitions = len(partitions)
        print(f"Number of partitions: {num_partitions}")
        # num_nodes = len(np.unique(list(partition_map.keys())))

        node2id = dict({node: idx for idx, node in enumerate(G.nodes())})
        id2node = dict(enumerate(node2id.keys()))
        comm2id = dict({community: idx for idx, community in enumerate(set(partition_map.values()))})
        id2comm = dict(enumerate(comm2id.keys()))
        # partition_matrix = np.zeros((num_nodes, num_partitions))
        partition_map = {node2id[node]: comm2id[community] for node, community in partition_map.items()}
        partition_map_copy = partition_map.copy()
        initial_labels = np.array(list(partition_map.values()))
        cnt = 0
        G = nx.relabel_nodes(G, node2id)
        A = np.array(nx.adjacency_matrix(G).todense())

        # while True:
        #     had_change = False
        for node, community in np.random.permutation(list(partition_map_copy.items())):
            partition_counts = defaultdict(float)
            for adjacent_node in G[node]:
                partition_counts[partition_map_copy[adjacent_node]] += A[node][adjacent_node]

            chosen = Counter(partition_counts).most_common(1)[0][0]
            if self.verbose: print(f"Node {node} moved: {community} -> {chosen}")
            if chosen != partition_map_copy[node]:
                had_change = True
            partition_map_copy[node] = chosen
        # cnt += 1
        curr_prt_num = len(np.unique(list(partition_map_copy.values())))
            # if had_change is False:
            #     print(f"BREAK: No change -> {had_change}!")
            #     break
            # if cnt > self.max_iter:
            #     print(f"BREAK: Max iteration reached {cnt}!")
            #     break
        if curr_prt_num <= 1:
            partition_map_copy = partition_map
            print(f"BREAK: Partition size is {curr_prt_num}!")

        new_labels = np.array(list(partition_map_copy.values()))
        changes = initial_labels != new_labels
        initial_labels = new_labels
        num_changes = changes.sum()

        resulting_map = {id2node[node]: community for node, community in partition_map_copy.items()}
        print(f"Number of changes {num_changes}")
        return resulting_map, num_changes

    def sub_divide(self, G, partition_map):
        num_changes = 0
        partition_map = {node: idx for idx, node in enumerate(partition_map)}
        partitions = np.unique(list(partition_map.values()))
        num_partitions = len(partitions)
        num_nodes = len(np.unique(list(partition_map.keys())))

        node2id = dict({node: idx for idx, node in enumerate(G.nodes())})
        id2node = dict(enumerate(node2id.keys()))
        comm2id = dict({community: idx for idx, community in enumerate(set(partition_map.values()))})
        id2comm = dict(enumerate(comm2id.keys()))
        partition_matrix = np.zeros((num_nodes, num_partitions))
        partition_map_copy = {node2id[node]: comm2id[community] for node, community in partition_map.items()}
        initial_labels = np.array(list(partition_map.values()))

        for node, community in np.random.permutation(list(partition_map_copy.items())):
            adj_partitions = [partition_map_copy[node2id[adjacent_node]] for adjacent_node in G[id2node[node]]]

            self_partition = [partition_map_copy[node]]
            chosen = np.random.choice(adj_partitions + self_partition)
            partition_map_copy[node] = chosen

        new_labels = np.array(list(partition_map_copy.values()))
        changes = initial_labels != new_labels
        initial_labels = new_labels
        num_changes = changes.sum()
        resulting_map = {id2node[node]: id2comm[community] for node, community in partition_map_copy.items()}

        return resulting_map

    def _randargmax(self, b, **kw):
        """ a random tie-breaking argmax"""
        return np.argmax(np.random.random(b.shape) * (b == b.max(axis=1)), **kw)
