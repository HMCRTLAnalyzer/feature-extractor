#!/bin/python3

# Written by Kaitlin Lucio, 2/25/23
# No License

# Import libraries

import networkx as nx
from networkx.algorithms import *
from pprint import *
import os
from datetime import datetime
import time
import logging

logger = logging.getLogger(__name__)

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


def logicalEffortEdgeMap(graph):
    """
        Takes in a directed acyclic graph and puts a numerical representation of the calculated logical effort on the 
        edges flowing toward the output.

        For more information on logical effort see: https://en.wikipedia.org/wiki/Logical_effort

        Input: graph
        Output: modified graph
    """
    for n in graph.nodes():
        add_edge_list = []
        if n[0] not in ["N", "I"]:
            parasitic_delay = 2
        else:
            parasitic_delay = 1
        effective_fanout = 0
        for p in graph.predecessors(n):
            if p[0] not in ["L","N"]:
                effective_fanout += (4/3)
            else:
                effective_fanout += 1
            add_edge_list = [(p, n)]
        node_delay = parasitic_delay + effective_fanout
        graph.add_edges_from(add_edge_list,weight=node_delay)

    return graph


def replaceBufWithLatch(graph):
    """
        Removes buffers inserted between inputs

        Input: graph
        Output: graph with buffers removed
    """
    original_graph = graph.copy()
    for n in original_graph.nodes():
        input = [s for s in original_graph.successors(n) if s[0] == "I" and s[1] != "L"]
        if input:
            logging.debug(f"Input {n} has buffers {input}")
            s = input[0]
            rm_edge_list = [(p, n) for p in original_graph.predecessors(n)]
            add_edge_list = [(p, s) for p in original_graph.predecessors(n)]
            graph.remove_edges_from(rm_edge_list)
            graph.add_edges_from(add_edge_list)
            graph.remove_node(n)

    return graph


def replaceEdgesWithNotNodes(graph):
    """
        Takes in a networkX graph of an AIGER file and inserts a node N_i between the successor and predecessor if there should be a NOT there.
            Case 1: NAND into not to get an AND gate (NAND is easier to analyze using logical effort)
            Case 2: NOT from a latch

        Only used on networkX graphs generated from the AIGER format.

        Input: networkX graph
        Output: copy of above networkX graph
    """
    original_graph = graph.copy()
    not_idx = 0
    for n in original_graph.nodes():
        rm_edge_list = []
        p_not_list = []
        for p in original_graph.predecessors(n):
            # Case 2: NOT from input
            if "I" in n:
                logging.debug(f"Predecessor {n} is an input")
                if original_graph[p][n]["arrowhead"] == "dot":
                    rm_edge_list += [(p,n)]
                    p_not_list += [p]
            # Case 1: All other cases
            else: 
                if original_graph[p][n]["arrowhead"] == "none":
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
    """
        Takes a path to an AIGER file as well as the input and output basenames (ex: I, O, L for AIGER) 
        and cuts it into a dictionary of graphs defined as follows
            graph_dict[i] = {"graph", "inputs", "outputs"}
    """
    
    start = time.time()

    original_graph = nx.DiGraph(nx.nx_pydot.read_dot(filepath))
    nodes = list(original_graph.nodes)

    graph_dict = {}
    input_names = []
    output_names = []
    i = 0

    # Replace top-level nodes which are latches with the name "latch". They are guaranteed to not have successors in this format

    # Get mapping of old nodes to new nodes
    mapping = {}

    latch_edge_rm = []

    for node in nodes:
        predecessors = getDiGraphPredecessors(original_graph, node)
        successors = getDiGraphSuccessors(original_graph, node)
        # Get latch name from predecessors
        latch_list = [x for x in predecessors if "L" in x and not successors]
        if latch_list:
            current_latch = latch_list.pop(0)
            latchName = f"I{current_latch}"
            latch_edge_rm += [(current_latch, latchName)]
            mapping[node] = latchName

    # Write changes found above, get new node mapping, and remove connections between latches
    original_graph = nx.relabel_nodes(original_graph, mapping, copy=False)
    original_graph.remove_edges_from(latch_edge_rm)
    nodes = list(original_graph.nodes)

    OL_found = []

    for node in nodes:

        # Reset output_names and input_names before scanning graph
        output_names = []
        input_names = []
        
        if any(x in node for x in output_basenames) and (node not in OL_found): # See if the node name contains any strings from the list for output/latch names
            output_names += [node]
            seen_nodes = [node]

            # Add found nodes to search_list for this node
            search_list = [node]
            
            # Iterate through search list after addin entries
            while search_list:
                current_nodes = search_list.pop(0)
                logging.debug(f"Searching adj for {current_nodes}")
                seen_nodes += [current_nodes]
                successors = getDiGraphSuccessors(original_graph, current_nodes)
                predecessors = getDiGraphPredecessors(original_graph, current_nodes)
                adj = list(set(successors + predecessors))
                logging.debug(f"Adjacent nodes to {current_nodes} are: {adj}")

                # Add nodes not seen before into list of future search nodes
                new_nodes = [x for x in adj if (x not in seen_nodes) and (x not in search_list) and not any(y in x for y in IO_basenames)]
                other_nodes = [x for x in adj if any(y in x for y in IO_basenames)]
                logging.debug(f"Saw new nodes {new_nodes} and {other_nodes}")
        
                # When we find a terminating node, add it to the list of found terminating nodes and ignore them for the future
                output_names += [x for x in other_nodes if any(y in x for y in output_basenames) and x[0] != "I"]
                if output_names:
                    logging.debug(f"Adding new output names {output_names}")
                seen_nodes = list(set(seen_nodes + other_nodes))

                # Get nodes who match IO_basenames
                search_list += new_nodes

            # Create graph using connected nodes
            logging.debug(f"Created subgraph using {seen_nodes}")
            sub_graph = original_graph.subgraph(seen_nodes).copy()

            # From seen nodes, grab input nodes
            for node in seen_nodes:
                if any(x in node for x in IO_basenames) and (node[0] == "I"):
                    input_names += [node]

            # Store found graph in graph_list under name of starter node, and store found outputs in output_names
            graph_dict[str(i)] = {"graph": sub_graph, "inputs": list(set(input_names)), "outputs": list(set(output_names))}
            OL_found += output_names + input_names

            # Increase iterator by 1

            i += 1
    
    end = time.time()
    logger.info(f"Time to cut {filepath} into DAGs: {end - start}s")

    return graph_dict


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

    logger.info(f"DOT file for {module} generated at {dot_filepath}")

    return dot_filepath

def getNodesOnPath(graph, output_list):
    """
        Takes in a graph and returns a dictionary with keys that point to a list of connected nodes

        Algorithm for searching modified from https://networkx.org/nx-guides/content/algorithms/dag/index.html.

        Searches a list of 
    """
    if not graph.is_directed():
        raise nx.NetworkXError("Topological sort not defined on undirected graphs.")

    # Get top-level nodes
    indegree_map = {n: d for n, d in graph.in_degree() if d > 0}
    zero_indegree = [v for v, d in graph.in_degree() if d == 0]
    node_mapping = {n: set(*()) for n, d in graph.in_degree() if d > 0}
    node_dict = {v: {v} for v in zero_indegree}
    logger.debug(f"Initialized zero_indegree: {zero_indegree}")
    logger.debug(f"Initialized empty sets for node mapping: {node_mapping}")

    while zero_indegree:
        this_generation = zero_indegree
        zero_indegree = []
        for node in this_generation:
            for child in graph.neighbors(node):
                # Create mapping for child nodes to map to upstream outputs
                if node in output_list:
                    logging.debug(f"Initializing mapping for {child} to {node}")
                    node_mapping[child] |= {node}
                else:
                    logging.debug(f"Adding existing mapping {node_mapping[node]} to {child}")
                    node_mapping[child] |= node_mapping[node]
                
                for output_node in node_mapping[child]:
                    node_dict[output_node] |= {child}

                indegree_map[child] -= 1
                if indegree_map[child] == 0:
                    logging.debug(f"Removed {child} from indegree mapping")
                    zero_indegree.append(child)
                    del indegree_map[child]
    logger.debug(f"Found {node_dict} while doing getNodesOnPath")

    if indegree_map:
        raise nx.NetworkXUnfeasible(
            "Graph contains a cycle or graph changed during iteration"
        )
    return node_dict




def getLogicalDepthAllPaths(graph, input_names, output_names):
    """
        Takes an list of inputs (leaves), list of outputs (ends) and returns the max length between each input and output

        Assumes the graph is structured such that outputs flow to inputs
    """
    logical_depth_lengths = []
    node_dict = getNodesOnPath(graph, output_names)
    for keys, entry in node_dict.items():
        try:
            logger.debug(f"Getting longest path length for {keys} with nodes {entry}")
            logical_depth_lengths += [len(dag_longest_path(graph.subgraph(entry)))]
        except Exception as e:
            logger.warn(e)

    return logical_depth_lengths


def topNLargestLD(graph_dict, N):
    """
        Takes in a graph dictionary of entries containing a graph, list of inputs, a list of outputs and returns the top N paths

        If N = 0, return entire list

        If N < 0, function breaks.
    """
    start = time.time()
    list_of_LDs = []
    for keys, entry in graph_dict.items():
        graph = entry["graph"]
        input_names = entry["inputs"]
        output_names = entry["outputs"]
        list_of_LDs += getLogicalDepthAllPaths(graph, input_names, output_names)
    list_of_LDs.sort(reverse=True)
    if N != 0:
        if len(list_of_LDs) >= N:
            largest_LD = list_of_LDs[0:N]
        else:
            largest_LD = list_of_LDs + [0]*(N-len(list_of_LDs))
    else:
        return list_of_LDs
    
    end = time.time()
    logger.info(f"Took {end - start}s to calculate Logical Effort Stats")

    return largest_LD

def getLongestLengthAllPaths(graph, input_names, output_names):
    """
        Takes an list of inputs (leaves), list of outputs (ends) and returns the max length between each input and output

        Assumes the graph is structured such that outputs flow to inputs
    """
    LE_path_lengths = []
    node_dict = getNodesOnPath(graph, output_names)
    for keys, entry in node_dict.items():
        try:
            logger.debug(f"Getting longest path length for {keys} with nodes {entry}")
            LE_path_lengths += [dag_longest_path_length(graph.subgraph(entry))]
        except Exception as e:
            logger.warn(e)

    return LE_path_lengths


def topNLargestLEandLD(graph_dict, N):
    """
        Takes in a graph dictionary of entries containing a graph, list of inputs, a list of outputs and returns the top N Logical Efforts
        as well as a list of Logical Efforts normalized by the graph's node count.
        If N = 0, return entire list

        If N < 0, function breaks.
    """
    start = time.time()

    list_of_LDs = []
    list_of_LEs = []
    list_of_LE_norm = []
    for keys, entry in graph_dict.items():
        graph = entry["graph"]
        input_names = entry["inputs"]
        output_names = entry["outputs"]
        NC = graph.order()
        list_of_LDs += getLogicalDepthAllPaths(graph, input_names, output_names)
        graph_LEs = getLongestLengthAllPaths(graph, input_names, output_names)
        list_of_LEs += graph_LEs
        list_of_LE_norm += [x/NC for x in graph_LEs]
    list_of_LEs.sort(reverse=True)
    list_of_LDs.sort(reverse=True)
    list_of_LE_norm.sort(reverse=True)
    if N != 0:
        if len(list_of_LEs) >= N:
            largest_LD = list_of_LDs[0:N]
            largest_LE = list_of_LEs[0:N]
            largest_LE_norm = list_of_LE_norm[0:N]
        else:
            largest_LD = list_of_LDs + [0]*(N-len(list_of_LDs))
            largest_LE = list_of_LEs + [0]*(N-len(list_of_LEs))
            largest_LE_norm = list_of_LE_norm + [0]*(N-len(list_of_LE_norm))
    else:
        return list_of_LEs, list_of_LE_norm, list_of_LDs
    
    end = time.time()
    logger.info(f"Took {end - start}s to calculate Logical Effort Stats")
    
    return largest_LE, largest_LE_norm, largest_LD

def topNCLBNodeCount(graph_dict,N):
    """
        Takes in a dictionary of graphs and returns the node count of each graph

        If N = 0, return all node counts

        N < 0 breaks this function
    """
    start = time.time()

    list_of_nodecounts = []
    for keys, entry in graph_dict.items():
        graph = entry["graph"]
        list_of_nodecounts += [graph.order()]
    list_of_nodecounts.sort(reverse=True)
    if N != 0:
        if len(list_of_nodecounts) >= N:
            list_of_nodecounts = list_of_nodecounts[0:N]
        else:
            list_of_nodecounts = list_of_nodecounts + [0]*(N-len(list_of_nodecounts))
    else:
        return list_of_nodecounts

    end = time.time()
    logger.info(f"Took {end - start}s to calculate Node Count Stats")

    return list_of_nodecounts

def fanoutList(graph):
    """
        Helper function for topNFanouts

        Returns a list of all the fanouts for a given graph
    """
    fanouts = []
    nodes = graph.nodes()
    for n in nodes:
        fanouts += [graph.in_degree(n)]
    return fanouts

def topNFanouts(graph_dict,N):
    """
        Takes in a dictionary of graphs and returns the largest fanouts of the graphs combined

        If N = 0, return all node counts

        N < 0 breaks this function
    """
    start = time.time()

    list_of_fanouts = []
    for keys, entry in graph_dict.items():
        graph = entry["graph"]
        list_of_fanouts += fanoutList(graph)
    list_of_fanouts.sort(reverse=True)
    if N != 0:
        if len(list_of_fanouts) >= N:
            list_of_fanouts = list_of_fanouts[0:N]
        else:
            list_of_fanouts = list_of_fanouts + [0]*(N-len(list_of_fanouts))
    else:
        return list_of_fanouts
    
    end = time.time()
    logger.info(f"Took {end - start}s to calculate Fanout Stats")

    return list_of_fanouts

def topNLE_NodeCountNormalized(graph_dict,N):
    """
        Takes in a graph dictionary of entries containing a graph, list of inputs, a list of outputs and returns the top N paths

        If N = 0, return entire list

        If N < 0, function breaks.
    """
    start = time.time()

    list_of_LE_norm = []
    for keys, entry in graph_dict.items():
        graph = entry["graph"]
        input_names = entry["inputs"]
        output_names = entry["outputs"]
        NC = graph.order()
        list_of_LEs = getLongestLengthAllPaths(graph, input_names, output_names)
        list_of_LE_norm += [x/NC for x in list_of_LEs]
    list_of_LEs.sort(reverse=True)
    if N != 0:
        if len(list_of_LEs) >= N:
            largest_LE = list_of_LE_norm[0:N]
        else:
            largest_LE = list_of_LE_norm + [0]*(N-len(list_of_LE_norm))
    else:
        return list_of_LEs
    
    end = time.time()
    logger.info(f"Took {end - start}s to calculate Logical Effort over Node Count")
    
    return largest_LE
