import networkx as nx
from networkx.algorithms import community as algorithms
from networkx.generators import community as generator
from networkx.algorithms.community.quality import modularity
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score
import math
import itertools
from collections import OrderedDict, Counter, deque, defaultdict
import pandas as pd
import multiprocess as mp
import matplotlib.cm as cm
import community as community_louvain
import scipy
from random import random
import operator
from helper.utils import extract_partition_map, extract_community_map
import io
import pickle
import torch
from torch import FloatTensor, LongTensor
from typing import Dict, Callable, List
from algorithms.louvain_core import LouvainCoreAlgorithm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# log_file = io.open("log.txt", "w")
log_file_2 = io.open("log2.txt", "w")


class GloveMaximizationAlgorithm(LouvainCoreAlgorithm):
    walk_length = None
    level_word_vectors = None
    mode = None

    def __init__(self, max_iter=20, fitness_function=community_louvain.modularity, verbose=False, max_local_movements=100, mode=-1, walk_length=int(1e5), stop_below=0.0001):
        self.walk_length = walk_length
        self.mode = mode
        self.stop_below = stop_below
        super().__init__(max_iter=max_iter, fitness_function=fitness_function, verbose=verbose, max_local_movements=max_local_movements)

    def initialize(self, G):
        initial_partition_map = dict(enumerate(G.nodes()))
        self.levels = []
        self.level_word_vectors = []
        self.stats = {"local_moving": [], "cooccurence_matrices": [], "partision_diff_sums": []}
        self.levels.append(initial_partition_map)
        self.level_fitness.append(0)
        self.level_graphs.append(G)
        self.gain_stats = []
        self.build_model(G)
        return G, initial_partition_map

    def local_movement(self, G, partition_map):
        # print(G.nodes())

        if len(G.nodes()) <= 1:
            return partition_map, 0
        num_changes = 0
        cnt = 0
        node2id = dict({node: idx for idx, node in enumerate(G.nodes())})
        id2node = dict(enumerate(node2id.keys()))
        comm2id = dict({community: idx for idx, community in enumerate(set(partition_map.values()))})
        partition_map_copy = {node2id[node]: comm2id[community] for node, community in partition_map.items()}
        resulting_map = partition_map_copy.copy()

        G = nx.relabel_nodes(G, node2id)
        A = nx.adjacency_matrix(G).todense()

        word_vectors = self.level_word_vectors[-1]
        curr_overall_differences, partition_gains = self._compute_overall_differences({node: 0 for node in partition_map_copy}, word_vectors)
        last_diff = 1000
        # initial run
        if self.mode == 0:
            # Random Order with closest neighbor initialisation
            random_node_order = np.random.permutation(list(partition_map_copy.keys()))
            for node_idx in random_node_order:
                adj_nodes = list(G[node_idx])
                distances = [(adj_node, torch.dist(word_vectors[node_idx], word_vectors[adj_node]), partition_map_copy[adj_node]) for adj_node in adj_nodes]
                closest_node = min(distances, key=operator.itemgetter(1))
                closest_node_idx, node_distance, new_prt = closest_node
                partition_map_copy[node_idx] = new_prt
        elif self.mode == 1:
            # Random Order with farthest neighbor initialization
            random_node_order = np.random.permutation(list(partition_map_copy.keys()))
            for node_idx in random_node_order:
                adj_nodes = list(G[node_idx])
                distances = [(adj_node, torch.dist(word_vectors[node_idx], word_vectors[adj_node]), partition_map_copy[adj_node]) for adj_node in adj_nodes]
                closest_node = max(distances, key=operator.itemgetter(1))
                closest_node_idx, node_distance, new_prt = closest_node
                partition_map_copy[node_idx] = new_prt
        elif self.mode == 2:
            # Random Order with Random neighbor partition assignment
            random_node_order = np.random.permutation(list(partition_map_copy.keys()))
            for node_idx in random_node_order:
                adj_node = np.random.choice(list(G[node_idx]))
                new_prt = partition_map_copy[adj_node]
                partition_map_copy[node_idx] = new_prt

        nodes = list(G.nodes())
        partition_gains = defaultdict(float)
        prev_sum_gains = self._L2_distance(nodes, word_vectors) * len(nodes)
        prev_sum_gains = 0
        prev_sum_gains, partition_gains = self._compute_overall_differences(partition_map_copy, word_vectors)
        prev_iteration_sum = 0
        log_file_2.write(f"\n--- START Sum: {prev_sum_gains:.10f}---\n")
        while True:
            rollier = deque(maxlen=10)
            current_communities = np.unique(list(partition_map_copy.values()))
            random_node_order = np.random.permutation(list(partition_map_copy.keys()))
            for node_idx in random_node_order:
                curr_prt = partition_map_copy[node_idx]

                prt_candidates = set(partition_map_copy[adj_node] for adj_node in list(G[node_idx])
                                     #   if partition_map_copy[adj_node] != curr_prt
                                     )

                empty_community = next(iter(set(range(min(current_communities), max(current_communities) + 2)) - set(current_communities)))
                prt_candidates.add(empty_community)
                prt_nodes = [node for node, community in partition_map_copy.items() if community == curr_prt and node != node_idx]
                curr_avg_diff = self._L2_distance(prt_nodes + [node_idx], word_vectors)
                curr_avg_diff_with_change = self._L2_distance(prt_nodes, word_vectors)
                giver_normalizer = len(prt_nodes)
                giver_gain = (curr_avg_diff_with_change - curr_avg_diff).cpu().numpy() / giver_normalizer if len(prt_nodes) < 1 else 0.00001

                change_candidates = []
                for idx, prt_candidate in enumerate(prt_candidates):

                    prt_candidate_nodes = [node for node, community in partition_map_copy.items() if community == prt_candidate]
                    candidate_avg_diff = self._L2_distance(prt_candidate_nodes, word_vectors)
                    candidate_avg_diff_with_change = self._L2_distance(prt_candidate_nodes + [node_idx], word_vectors)
                    receiver_gain = (candidate_avg_diff - candidate_avg_diff_with_change).cpu().numpy() if len(prt_candidate_nodes) > 0 else 0.00001
                    receiver_normalizer = sum(np.diagonal(A)[prt_candidate_nodes])
                    # receiver_normalizer = 1
                    if receiver_normalizer == 0:
                        receiver_normalizer = 1

                    # change_gain = (receiver_gain / receiver_normalizer)
                    change_gain = (receiver_gain + giver_gain / receiver_normalizer)

                    difference_increase = np.sum(list(partition_gains.values())) - giver_gain + receiver_gain
                    change_candidates.append((prt_candidate, change_gain, receiver_gain, giver_gain, receiver_normalizer, change_gain / difference_increase))

                choose = 1
                maximum_gain = max(change_candidates, key=operator.itemgetter(choose))

                new_prt, abs_gain, receiver_gain, giver_gain, receiver_normalizer, difference_increase = maximum_gain
                partition_map_copy[node_idx] = new_prt



                rollier.append(abs_gain)
                rolling_mean = np.mean(rollier)

                data_point = {
                    "rol": rolling_mean,
                    "receiver_gain": receiver_gain,
                    "giver_gain": giver_gain,
                    "normalizer": receiver_normalizer,
                    "abs": abs_gain,
                    "candidates": len(partition_map_copy),
                    "partitions": len(set(partition_map_copy.values())),
                }
                self.stats["local_moving"].append(data_point)
                num_changes += 1

            curr_overall_differences, partition_gains = self._compute_overall_differences(partition_map_copy, word_vectors)
            log_file_2.write(f"Sum : {curr_overall_differences:.10f} | Diff: {curr_overall_differences - prev_sum_gains:.10f}\n")
            num_remaining_partitions = len(set(partition_map_copy.values()))
            iter_diff = np.abs(curr_overall_differences - prev_iteration_sum)
            print(f"Num Partitions {num_remaining_partitions} | Sum {curr_overall_differences} | Diff {np.abs(iter_diff)} | Last {np.abs(last_diff - iter_diff)}")
            if iter_diff < self.stop_below:
                print(f"BREAK: No difference from last run: {iter_diff}:{last_diff}")
                # resulting_map = partition_map_copy.copy()
                break
            if num_remaining_partitions == 1 and not self.mode == 3:
                print(f"BREAK: Only one partition remained: {num_remaining_partitions}")
                resulting_map = partition_map
                break
            # if np.abs(iter_diff) < .1:
            #     print(f"BREAK: Difference is: {iter_diff} | Abs: {np.abs(iter_diff)}")
            #     resulting_map = partition_map_copy.copy()
            #     break
            resulting_map = partition_map_copy.copy()
            prev_iteration_sum = curr_overall_differences
            last_diff = iter_diff

            cnt += 1
            if self.mode == 3:
                print("Mode 3 activated")
                break
            # if
            if cnt >= self.max_iter:
                print(f"Max iteration reached: {cnt}")
                break

        comm2id = dict({community: idx for idx, community in enumerate(set(resulting_map.values()))})
        resulting_map = {id2node[node]: comm2id[community] for node, community in resulting_map.items()}
        # log_file.write(f"Done! Number of changes {num_changes}\n")
        print(f"Number of changes {num_changes}")

        self.rebuild_word_vectors(resulting_map, word_vectors)
        return resulting_map, curr_overall_differences

    def choose_right_candidate(self, change_candidates, partition_differences):
        pass

    def _compute_overall_differences(self, partition_map, word_vectors):
        communities = set(partition_map.values())
        new_word_vectors = np.zeros(len(communities))
        values = defaultdict(float)
        for idx, community in enumerate(communities):
            community_members = [node for node, prt in partition_map.items() if prt == community]
            community_distance = self._L2_distance(community_members, word_vectors).cpu().numpy().sum()
            values[community] = community_distance
            new_word_vectors[idx] = community_distance

        return new_word_vectors.sum(), values

    def rebuild_word_vectors(self, partition_map, word_vectors):
        communities = set(partition_map.values())
        new_word_vectors = torch.zeros([len(communities), word_vectors.shape[1]])
        for community in communities:
            community_members = [node for node, prt in partition_map.items() if prt == community]
            prt_candidate_vectors = word_vectors[community_members]
            prt_candidate_centroid = prt_candidate_vectors.sum(axis=0) / prt_candidate_vectors.shape[0]
            new_word_vectors[community, :] = prt_candidate_centroid

        self.level_word_vectors.append(new_word_vectors)

    def build_model(self, G):
        sliding_windows, _, _ = self._sample_graph(G)
        cooccurence_matrix = self._create_coorccurence_matrix(sliding_windows, G)
        X = torch.tensor(np.power(cooccurence_matrix, 2))
        X = X.to(device)  # If GPU enabled
        X = X.to(torch.float64) + 0.0001
        X_weighted = self._weight_fn(X, 100, 0.75)
        X_weighted = X_weighted.to(device)
        word_vectors = self._train(X=X, X_weighted=X_weighted)
        self.level_word_vectors.append(word_vectors)
        return word_vectors

    def _random_walk(self, a, i, iters):
        # a -> adj
        # i -> starting row
        walk = np.zeros(iters + 1)  # holds transitions
        walk[0] = i
        elements = np.arange(a.shape[0])  # for our graph [0,1,2,3]
        c_index = i  # current index for this iteration
        for k in range(iters):
            probs = a[c_index].reshape((-1, ))  # probability of transitions
            # sample from probs
            sample = np.random.choice(elements, p=probs)  # sample a target using probs
            index = sample  # go to target
            walk[k + 1] = index
            c_index = index
        return walk

    def _sample_graph(self, G):

        G.remove_edges_from(nx.selfloop_edges(G))  # TODO: Implementation detail
        walk_length = self.walk_length
        markov_matrix = np.array(nx.google_matrix(G, alpha=1))
        nodes = G.nodes()
        vocab = {f"node_{node}": node for node in nodes}
        n2voc = {node: name for name, node in vocab.items()}
        starting_point = np.random.choice(nodes)
        walk = self._random_walk(markov_matrix, starting_point, walk_length)
        sliding_windows = np.vstack((walk, np.roll(walk, -1), np.roll(walk, -2), np.roll(walk, -3), np.roll(walk, -4))).astype(np.int)
        return sliding_windows, vocab, n2voc

    def _create_coorccurence_matrix(self, sliding_windows, G):
        cooccurence_matrix = np.zeros_like(nx.adjacency_matrix(G).todense())

        for position in range(self.walk_length):
            left_left_word = sliding_windows[0, position]
            left_word = sliding_windows[1, position]
            center_word = sliding_windows[2, position]
            right_word = sliding_windows[3, position]
            right_right_word = sliding_windows[4, position]
            # print(cooccurence_matrix.shape)
            cooccurence_matrix[center_word, left_word] += 1
            cooccurence_matrix[center_word, right_word] += 1
            cooccurence_matrix[center_word, left_left_word] += 1
            cooccurence_matrix[center_word, right_right_word] += 1

        self.stats["cooccurence_matrices"].append(pd.DataFrame(cooccurence_matrix))
        return cooccurence_matrix

    def _train(self, X_weighted, X):
        num_epochs = 300
        all_losses = []
        num_nodes = X_weighted.shape[0]
        network = GloVe(num_nodes=num_nodes, vector_dimensionality=30, device=device)
        opt = torch.optim.Adam(network.parameters(), lr=0.05)

        for i in range(num_epochs):
            loss = network.forward(X_weighted, X)  # backward
            value = loss.data.cpu().numpy()
            if i % 20 == 0:
                print(f"Epoch: {i} - Loss is currently at: {loss}")
            all_losses.append(value)
            loss.backward()
            opt.step()
            opt.zero_grad()

        word_vectors = network.get_vectors().detach()
        return word_vectors

    def _weight_fn(self, X: FloatTensor, x_max: int, a: float) -> FloatTensor:
        result = torch.where(X < x_max, torch.pow(X / x_max, 0.75), torch.ones_like(X))
        return result

    def _cosine_similarity(self, nodes, word_vectors):
        if len(nodes) == 0:
            return torch.zeros_like(word_vectors[[0]]).sum()
        prt_candidate_vectors = word_vectors[nodes]
        prt_candidate_centroid = prt_candidate_vectors.sum(axis=0) / prt_candidate_vectors.shape[0]
        comparison_vector = prt_candidate_centroid
        inner_products = torch.mm(comparison_vector.view(1, -1), prt_candidate_vectors.transpose(0, 1))
        matrix_norms = torch.norm(prt_candidate_vectors, p=2, dim=1)
        comparison_norm = torch.norm(comparison_vector, p=2)
        candidate_sum_of_centroid_diff = inner_products / (comparison_norm * matrix_norms)
        candidate_avg_centroid_differences = candidate_sum_of_centroid_diff.sum() / candidate_sum_of_centroid_diff.shape[1]
        return candidate_avg_centroid_differences

    def _L2_distance(self, nodes, word_vectors):
        if len(nodes) <= 1:
            return torch.zeros_like(word_vectors[[0]]).sum()
        prt_candidate_vectors = word_vectors[nodes]
        prt_candidate_centroid = prt_candidate_vectors.sum(axis=0) / prt_candidate_vectors.shape[0]
        candidate_sum_of_centroid_diff = torch.dist(prt_candidate_centroid, prt_candidate_vectors)
        candidate_avg_centroid_differences = candidate_sum_of_centroid_diff.sum() / len(nodes)
        return candidate_avg_centroid_differences


class GloVe(torch.nn.Module):
    def __init__(self, num_nodes: int, vector_dimensionality: int = 30, device: str = 'cpu') -> None:
        super(GloVe, self).__init__()
        self.device = device
        self.vocab_len = num_nodes
        self.w = torch.nn.Embedding(num_embeddings=self.vocab_len, embedding_dim=vector_dimensionality).to(self.device)
        self.wc = torch.nn.Embedding(num_embeddings=self.vocab_len, embedding_dim=vector_dimensionality).to(self.device)
        self.b = torch.nn.Embedding(num_embeddings=self.vocab_len, embedding_dim=1).to(self.device)
        self.bc = torch.nn.Embedding(num_embeddings=self.vocab_len, embedding_dim=1).to(self.device)

    def forward(self, X_weighted: FloatTensor, X: FloatTensor) -> FloatTensor:
        embedding_input = torch.arange(self.vocab_len).to(self.device)
        W = self.w(embedding_input)
        W_context = self.wc(embedding_input)
        B = self.b(embedding_input)
        B_context = self.bc(embedding_input)
        return self._loss_fn(X_weighted, W, W_context, B, B_context, X, self.device)

    def get_vectors(self) -> FloatTensor:
        embedding_input = torch.arange(self.vocab_len).to(self.device)
        return self.w(embedding_input) + self.wc(embedding_input)

    def _loss_fn(self, X_weighted: FloatTensor, W: FloatTensor, W_context: FloatTensor, B: FloatTensor, B_context: FloatTensor, X: FloatTensor, device: str = "cpu") -> FloatTensor:
        hypothesis = (torch.mm(W, W_context.transpose(0, 1)) + B + B_context).type(torch.DoubleTensor).to(device)
        target = torch.log(X)
        squared_loss = ((hypothesis - target)**2)
        temp = torch.mul(X_weighted, squared_loss)
        result = torch.sum(temp)
        return result
