#   -----------------------------------------   #
#   Script to generate dot files from general-synth results   
#   Written by Kaitlin Lucio 11/27/23
#
#   -----------------------------------------   #

# Import libraries

import networkx as nx

# Functions

def read_dot(filepath):
    """
        Function to read a graph from a dot file

        Input: path to dot file
        Output: graph object from dot file
    """
    return nx.read_dot(filepath)

def create_dag(graph, input_list, output_list):
    """
        Function that checks a graph is a directed acyclic graph and cuts down the IO until it becomes a DAG

        Input: networkx graph, inputs, outputs
        Output: smaller networkx graph
    """
    cut_graph = graph

    return cut_graph


