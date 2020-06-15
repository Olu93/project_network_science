from algorithms.louvain_core import LouvainCoreAlgorithm
from algorithms.map_equation import map_equation_essentials, compute_minimal_codelength, retrieve_linkings, map_equation, aggregate_in_ex_nodes, map_equation_old
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
        # print(partition_map)
        unique_partitions = np.unique(list(partition_map.values()))
        # num_partitions = len(unique_partitions)
        cnt = 0
        node2id = dict({node: idx for idx, node in enumerate(G)})
        id2node = dict(enumerate(node2id.keys()))
        comm2id = dict({community: idx for idx, community in enumerate(set(partition_map.values()))})
        partition_map_copy = partition_map.copy()
        partition_map = {node2id[node]: comm2id[community] for node, community in partition_map.items()}
        # original_community_map = dict(enumerate(extract_community_map(partition_map)))  # For some reason partition zero misses
        # community_map = {idx: [node2id[node] for node in community] for idx, community in original_community_map.items()}
        if len(G.nodes()) <= 2:
            return partition_map_copy, 1
        G = nx.relabel_nodes(G, node2id)

        # print("Step 1")
        node_partition_in_links, node_partition_ex_links, node_weights, node_partitions, A_prt, A = map_equation_essentials(G, partition_map)
        # print("Step 2")
        p_a_i, q_out_i, q_out, p_circle_i, p_u = retrieve_linkings(node_partition_in_links, node_partition_ex_links, node_weights, node_partitions)
        # print("Step 3")
        L, index_codelength, module_codelength = compute_minimal_codelength(p_a_i, q_out_i, q_out, p_circle_i, p_u, node_partitions)

        # print("Step 4")
        print("START!")
        # print(partition_map_copy)
        # print(node_partition_in_links)
        # print(node_partition_ex_links)
        # L_check, _, _ = map_equation(G, partition_map)
        # L_check, _, _ = map_equation_old(G, partition_map)
        # L_check_2, _, _ = map_equation(G, partition_map)
        print("Compare")
        print(L)
        # print(L_check)
        # print(L_check_2)
        print("")
        last_iter_L = L
        while True:
            print(f"Starting search...")
            random_order = np.random.permutation(G.nodes())
            had_improvement = False
            for node in random_order:
                change_candidates = []
                node_prt = partition_map[node]
                current_communities = set(partition_map.values())
                empty_community = next(iter(set(range(min(current_communities), max(current_communities) + 2)) - set(current_communities)))
                adj_prt_candiates = set([
                    partition_map[adj] for adj in G[node]
                    # if partition_map[adj] != node_prt
                ] + [empty_community])
                for adj_prt in adj_prt_candiates:
                    # if adj_prt == node_prt:
                    #     if self.verbose: print(f"{node}: Skip {node_prt} -> {adj_prt}")
                    #     continue
                    tmp_A_prt = A_prt.copy()
                    tmp_A_prt[tmp_A_prt[:, node] == node_prt, node] = adj_prt
                    # print(A_prt)
                    # print(tmp_A_prt)
                    tmp_node_in_links, tmp_node_ex_links, tmp_node_partitions, zm = aggregate_in_ex_nodes(A, tmp_A_prt)
                    # print(tmp_node_in_links)
                    # print(tmp_node_ex_links)
                    p_a_i, q_out_i, q_out, p_circle_i, p_u = retrieve_linkings(tmp_node_in_links, tmp_node_ex_links, node_weights, tmp_node_partitions)
                    new_L, new_index_codelength, new_module_codelength = compute_minimal_codelength(p_a_i, q_out_i, q_out, p_circle_i, p_u, tmp_node_partitions)
                    change_candidates.append((new_L, node, node_prt, adj_prt, tmp_A_prt))
                if len(change_candidates) == 0:
                    if self.verbose: print(f"{node}: No candidates available!")
                    continue

                chosen_change = min(change_candidates, key=operator.itemgetter(0))
                new_L, curr_node, curr_node_prt, new_node_prt, new_A_prt = chosen_change
                if L - new_L > 0.0000:
                    if self.verbose: print(f"Decrease in average code length! {L} > {new_L}")
                    partition_map[curr_node] = new_node_prt
                    A_prt = new_A_prt
                    had_improvement = True

                    L = new_L
                # L_check_2, _, _ = map_equation(G, partition_map)
                # L_check, _, _ = map_equation_old(G, partition_map)
                # print("Compare")
                # print(L)
                # print(L_check_2)
                # print(L_check)
                # print("")
            num_remaining_partitions = len(set(partition_map.values()))
            print(f"Remaining partitions: {num_remaining_partitions}")
            if num_remaining_partitions <= 2:
                print(f"BREAK: Too few partitions remained: {num_remaining_partitions}")
                return partition_map_copy, last_iter_L
            if last_iter_L - L < self.stop_below:
                print(f"BREAK: Improvement is marginal {last_iter_L} - {L} -> {last_iter_L - L} < {self.stop_below}")
                break
            if had_improvement is False:
                print(f"BREAK: No improvement happend {L}")
                break
            if cnt > self.max_iter:
                print(f"BREAK: Max iteration reached {cnt}")
                break
            last_iter_L = L
            cnt += 1
        resulting_map = {id2node[node]: community for node, community in partition_map.items()}
        print(f"Resulting code length {L}")
        return resulting_map, last_iter_L
