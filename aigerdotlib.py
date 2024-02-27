#!/bin/python3

# Written by Kaitlin Lucio, 2/25/23
# No License

# Import libraries

import networkx as nx
from networkx.algorithms import *
from pprint import *
import os
from datetime import datetime

def getDiGraphSuccessors(graph, node):
    """
        Gets all the nodes connected to the input node in a directed graph
        Inputs: graph [networkx digraph], node [name]
        Outputs: list of connected nodes
    """
    connected_nodes = []
    successors = graph.successors(node)
    for item in successors:
        connected_nodes += [item]
    return connected_nodes


def getDiGraphPredecessors(graph, node):
    """
        Gets all the nodes connected to the input node in a directed graph
        Inputs: graph [networkx digraph], node [name]
        Outputs: list of connected nodes
    """
    connected_nodes = []
    predecessors = graph.predecessors(node)
    for item in predecessors:
        connected_nodes += [item]
    return connected_nodes


def getDiGraphAdj(graph, node):
    connected_nodes = []
    successors = graph.successors(node)
    predecessors = graph.predecessors(node)
    for item in successors:
        connected_nodes += [item]
    for item in predecessors:
        predecessors += [item]
    return connected_nodes


def NotEdgetoNotNode(graph):
    """
        Takes in a networkX graph with dot heads as edges and inserts a node N_i between the successor and predecessor

        Only used on networkX graphs generated from the AIGER format.

        Input: networkX graph
        Output: copy of above networkX graph
    """
    original_graph = graph.copy()
    not_idx = 0
    rm_edge_list = []
    for n in original_graph.nodes():
        p_not_list = []
        for p in original_graph.predecessors(n):
            if original_graph[p][n]["arrowhead"] == "dot":
                rm_edge_list += [(p,n)]
                p_not_list += [p]
        if rm_edge_list:
            N_name = f"N{not_idx}"
            graph.remove_edges_from(rm_edge_list)
            graph.add_node(N_name)
            graph.add_edge(N_name,n, arrowhead="none")
            graph.add_edges_from([(p, N_name) for p in p_not_list])
            not_idx += 1

    return graph


def cutAIGERtoDAGs(filepath, IO_basenames, output_basenames):

    original_graph = nx.DiGraph(nx.nx_pydot.read_dot(filepath))
    nodes = list(original_graph.nodes)

    graph_list = []
    input_names = []
    output_names = []

    # Replace top-level nodes which are latches with the name "latch". They are guaranteed to not have successors in this format

    # Get mapping of old nodes to new nodes
    mapping = {}

    for node in nodes:
        predecessors = getDiGraphPredecessors(original_graph, node)
        successors = getDiGraphSuccessors(original_graph, node)
        # Get latch name from predecessors
        latch_list = [x for x in predecessors if "L" in x and not successors]
        if latch_list:
            latchName = f"I{latch_list.pop(0)}"
            mapping[node] = latchName

    # Write changes found above
    original_graph = nx.relabel_nodes(original_graph, mapping, copy=False)

    OL_found = []

    for node in nodes:
        if any(x in node for x in IO_basenames) and (node[0] == "I"):
            input_names += node

        if any(x in node for x in output_basenames) and (node not in OL_found): # See if the node name contains any strings from the list for output/latch names
            seen_nodes = [node]
            successors = original_graph.successors(node) # get nodes which feed into this latch/output

            # Add found nodes to search_list for this node
            search_list = [node]
            
            # Iterate through search list after addin entries
            while search_list:
                current_nodes = search_list.pop(0)
                seen_nodes += [current_nodes]
                successors = getDiGraphSuccessors(original_graph, current_nodes)
                predecessors = getDiGraphPredecessors(original_graph, current_nodes)
                adj = list(set(successors + predecessors))

                # Add nodes not seen before into list of future search nodes
                new_nodes = [x for x in adj if (x not in seen_nodes) and (x not in search_list) and not any(y in x for y in IO_basenames)]
                other_nodes = [x for x in adj if any(y in x for y in IO_basenames)]

                # When we find a terminating node, add it to the list of found terminating nodes and ignore them for the future
                OL_found += [x for x in other_nodes if any(y in x for y in output_basenames) and x[0] != "I"]
                seen_nodes = list(set(seen_nodes + other_nodes))

                # Get nodes who match IO_basenames
                predecessors = [x for x in predecessors]
                adj = [x for x in adj]
                search_list += new_nodes

            # Create graph using connected nodes
            sub_graph = original_graph.subgraph(seen_nodes).copy()

            # Store found graph in graph_list under name of starter node, and store current node in output_names
            output_names += [node]
            graph_list += {sub_graph}
    
    return graph_list, output_names, input_names


def generateDOT(srcdir, module, language):
    """
        Takes in a source directory, a (system)verilog module name, and a language. Returns the filepath of a .dot file generated using AIGER as an intermediate
        Calls yosys to perform synthesis, so could be quite expensive.

        inputs: source directory, module name, project language
        outputs: path to results
    """
    # DO NOT CHANGE ME WITHOUT CHANGING THE MAKEFILE
    results_path = f"./graphs/{module}"

    # Create path if it doesn't exist, rename existing files if directory exists
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    # Call makefile for project's verilog
    os.system(f"make aig_auto NAME={module} HDL_LANG={language} MODULE_NAME={module} SRC_PATH={srcdir}")

    # Generate path based on makefile
    dot_filepath = f"{results_path}/{module}.dot"

    return dot_filepath

