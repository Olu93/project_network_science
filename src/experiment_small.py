# %%
import numpy as np
import multiprocess as mp
import helper.utils as utils
import helper.quality_functions as qfunctions
import helper.visualization as viz
from algorithms.louvain_core import LouvainCoreAlgorithm
from algorithms.glove_louvain import GloveMaximizationAlgorithm
from algorithms.label_propagation_louvain import HierarchicalLabelPropagation
from algorithms.random_louvain import RandomPropagation
from algorithms.map_equation_louvain import MapEquationMaximization
import community as community_louvain
from networkx.algorithms import community as algorithms
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import pickle
from networkx.algorithms import community as algorithms
from networkx.generators import community as generator

# %%
if __name__ == "__main__":
    # %%
    G, pos = utils.generate_benchmark_graph(250, 0.1)
    true_partition, true_community = utils.extract_true_communities(G)
    # G = nx.barbell_graph(5, 2)
    # G = nx.karate_club_graph()
    # G = generator.planted_partition_graph(4, 25, p_in=0.9, p_out=0.1)
    # pos = nx.spring_layout(G)
    # true_partition = community_louvain.best_partition(G)
    algorithm = GloveMaximizationAlgorithm(fitness_function=None, verbose=False, max_iter=50)
    # algorithm = HierarchicalLabelPropagation(fitness_function=None, verbose=False, max_iter=50)
    # algorithm = RandomPropagation(fitness_function=None, verbose=False, max_iter=50)
    # algorithm = LouvainCoreAlgorithm(fitness_function=qfunctions.map_equation_wrapper_old, verbose=True, max_iter=50)
    # algorithm = MapEquationMaximization(fitness_function=None, max_iter=50, stop_below=0.0)
    partition = algorithm.run(G)
    # # partition = community_louvain.best_partition(G)
    # # %%
    viz.show_intermediate_results(algorithm, G, true_partition)
    plt.show()
    # viz.visualize_benchmark_graph(G, pos, partition)
    # plt.show()
    # # # %%
    viz.show_all_identified_partitions(G, pos, partition)
    plt.show()
    # pickle.dump(algorithm, open("last_run.pkl", "wb"))
    # %%
    # viz.show_reduction(algorithm, G, true_partition)
    # plt.show()

# # %%
# partition

# # %%
# qfunctions.map_equation_wrapper(partition, G)

# # %%
# qfunctions.map_equation_wrapper_old(partition, G)

# # %%
# qfunctions.map_equation(G, partition)

# # %%
# qfunctions.map_equation_old(G, partition)

# %%
