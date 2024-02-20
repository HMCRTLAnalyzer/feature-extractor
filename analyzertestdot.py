#!/bin/python3

# Import libraries

import networkx as nx

# Read in graph and print it 

# filepath = "decoder_latch.dot"
filepath = "graphs/testcyclicflop/testcyclicflop_ORIGINAL_2024-01-25-22-20-01/testcyclicflop_ORIGINAL.dot"
origGraph = nx.DiGraph(nx.nx_pydot.read_dot(filepath))
print(origGraph.is_directed())

nodes = list(origGraph.nodes)
print(nodes)

for node in nodes:
    print(node)
    if "c" in node:
        label = ""
        try:
            label = origGraph.nodes[node]["label"]
        except:
            pass
        # TODO: Figure out how to get grouping of nodes using networkx
        if "FF" in label:
            print(f"found flop at {node}")
            neighbors = origGraph.successors(node)
            listOfAdj = []
            for item in neighbors:
                listOfAdj += [item]
            print(listOfAdj)
            print(origGraph.adj[node])
            # print(node)
            # successors = origGraph.successors(node)
            # predecessors = origGraph.predecessors(node)
            # listOfAdj = []
            # for item in successors:
            #     listOfAdj += [item]
            # # for item in predecessors:
            # #     listOfAdj += [item]
            # print(listOfAdj)