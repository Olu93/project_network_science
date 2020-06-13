from algorithms.louvain_core import LouvainCoreAlgorithm
from algorithms.map_equation import map_equation_essentials, compute_minimal_codelength, retrieve_linkings, map_equation, aggregate_in_ex_nodes
import numpy as np
import networkx as nx
import operator
import warnings
warnings.filterwarnings("ignore")

class MapEquationMaximization(LouvainCoreAlgorithm):
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
        def compute_weighted_entropy(probs, normalizer):
            return np.nansum((probs / normalizer) * np.log2(probs / normalizer))

        def compute_partition_probability(prt_id, node_weights, partition_map):
            community = [node for node, prt in partition_map.items() if prt == prt_id]
            return node_weights[community].sum()

        unique_partitions = np.unique(list(partition_map.values()))
        # num_partitions = len(unique_partitions)

        node2id = dict({node: idx for idx, node in enumerate(G)})
        # id2node = dict(enumerate(node2id.keys()))
        comm2id = dict({community: idx for idx, community in enumerate(set(partition_map.values()))})
        partition_map = {node2id[node]: comm2id[community] for node, community in partition_map.items()}
        # original_community_map = dict(enumerate(extract_community_map(partition_map)))  # For some reason partition zero misses
        # community_map = {idx: [node2id[node] for node in community] for idx, community in original_community_map.items()}
        G = nx.relabel_nodes(G, node2id)

        print("Step 1")
        node_partition_in_links, node_partition_ex_links, node_weights, node_partitions, A_prt, A = map_equation_essentials(G, partition_map)
        print("Step 2")
        p_a_i, q_out_i, q_out, p_circle_i, p_u = retrieve_linkings(node_partition_in_links, node_partition_ex_links, node_weights, node_partitions)
        print("Step 3")
        L, index_codelength, module_codelength = compute_minimal_codelength(p_a_i, q_out_i, q_out, p_circle_i, p_u, node_partitions)

        print("Step 4")
        print("START!")
        print(node_partition_in_links)
        print(node_partition_ex_links)
        L_check, _, _ = map_equation(G, partition_map)
        print(L)
        print(L_check)
        print("")
        while True:
            random_order = np.random.permutation(G.nodes())
            had_improvement = False
            for node in random_order:
                change_candidates = []
                node_prt = partition_map[node]
                for adj in G[node]:
                    tmp_A_prt = A_prt.copy()
                    adj_prt = partition_map[adj]
                    print(f"{node}: {node_prt} -> {adj_prt}")
                    tmp_A_prt[tmp_A_prt[:, node] == node_prt, node] = adj_prt
                    # print(A_prt)
                    # print(tmp_A_prt)
                    tmp_node_in_links, tmp_node_ex_links, tmp_node_partitions, zm = aggregate_in_ex_nodes(A, tmp_A_prt)
                    # print(tmp_node_in_links)
                    # print(tmp_node_ex_links)
                    p_a_i, q_out_i, q_out, p_circle_i, p_u = retrieve_linkings(tmp_node_in_links, tmp_node_ex_links, node_weights, tmp_node_partitions)
                    new_L, new_index_codelength, new_module_codelength = compute_minimal_codelength(p_a_i, q_out_i, q_out, p_circle_i, p_u, tmp_node_partitions)
                    change_candidates.append((new_L, node, node_prt, adj_prt, tmp_A_prt))
                chosen_change = min(change_candidates, key=operator.itemgetter(0))
                new_L, curr_node, curr_node_prt, new_node_prt, new_A_prt = chosen_change
                L = new_L
                partition_map[curr_node] = new_node_prt
                A_prt = new_A_prt
                print("CHECK")
                # print(A_prt)
                L_check, _, _ = map_equation(G, partition_map)
                print("Compare")
                print(L)
                print(L_check)


        # resulting_map = {id2node[node]: id2comm[community] for node, community in partition_map_copy.items()}
        # print(f"Number of changes {num_changes}")
        # return resulting_map, num_changes
