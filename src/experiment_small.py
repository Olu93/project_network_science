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

# %%
if __name__ == "__main__":
    G, pos = utils.generate_benchmark_graph(250, 0.1)
    true_partition, true_community = utils.extract_true_communities(G)
    G = nx.barbell_graph(4, 1)
    pos = nx.spring_layout(G)
    # algorithm = GloveMaximizationAlgorithm(fitness_function=None, verbose=False, max_iter=50, mode=-1)
    # algorithm = HierarchicalLabelPropagation(fitness_function=None, verbose=False, max_iter=50)
    # algorithm = RandomPropagation(fitness_function=None, verbose=False, max_iter=50)
    # algorithm = LouvainCoreAlgorithm(fitness_function=qfunctions.random_wrapper, verbose=False, max_iter=50)
    algorithm = MapEquationMaximization(fitness_function=None, max_iter=50)
    partition = algorithm.run(G)
    # # partition = community_louvain.best_partition(G)
    # # %%
    viz.show_intermediate_results(algorithm, G, true_partition)
    plt.show()
    # viz.visualize_benchmark_graph(G, pos, partition)
    # plt.show()
    # # # %%
    # viz.show_all_identified_partitions(G, pos, partition)
    # plt.show()
    # pickle.dump(algorithm, open("last_run.pkl", "wb"))
    # %%
    # viz.show_reduction(algorithm, G, true_partition)
    # plt.show()

# %%
