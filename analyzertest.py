#!/bin/python3

# Import libraries

import networkx as nx
from networkx.algorithms import *
from pprint import *

# TODO: Move this to its own library file

def getDiGraphSuccessors(graph, node):
    """
        Gets all the nodes connected to the input node in a directed graph
        Inputs: graph [networkx digraph], node [name]
        Outputs: list of connected nodes
    """
    connectedNodes = []
    successors = graph.successors(node)
    for item in successors:
        connectedNodes += [item]
    return connectedNodes

def getDiGraphPredecessors(graph, node):
    """
        Gets all the nodes connected to the input node in a directed graph
        Inputs: graph [networkx digraph], node [name]
        Outputs: list of connected nodes
    """
    connectedNodes = []
    predecessors = graph.predecessors(node)
    for item in predecessors:
        connectedNodes += [item]
    return connectedNodes

def getDiGraphAdj(graph, node):
    connectedNodes = []
    successors = graph.successors(node)
    predecessors = graph.predecessors(node)
    for item in successors:
        connectedNodes += [item]
    for item in predecessors:
        predecessors += [item]
    return connectedNodes

# Define Latch, Output labels list
IOLList = ["L","O","I"]
OLList = ["L","O"]

# Read in graph and print its nodes

filepath = "basiccyclic.dot"
origGraph = nx.DiGraph(nx.nx_pydot.read_dot(filepath))
print(origGraph.is_directed())

nodes = list(origGraph.nodes)
print(origGraph.nodes(data="color")) # Print colors
print(origGraph)
print(nx.is_directed_acyclic_graph(origGraph))

# TODO: Figure out what needs to be turned into a function for creation

graphList = []
inputList = []
outputList = []

# Replace top-level nodes which are latches with the name "latch". They are guaranteed to not have successors in this format

# Get mapping of old nodes to new nodes
mapping = {}

for node in nodes:
    predecessors = getDiGraphPredecessors(origGraph, node)
    successors = getDiGraphSuccessors(origGraph, node)
    # Get latch name from predecessors
    latchList = [x for x in predecessors if "L" in x and not successors]
    if latchList:
        latchName = f"I{latchList.pop(0)}"
        mapping[node] = latchName

# Write changes found above
origGraph = nx.relabel_nodes(origGraph, mapping, copy=False)

OLFound = []

for node in nodes:

    if any(x in node for x in OLList) and (node not in OLFound): # See if the node name contains any strings from the list for output/latch names
        seenNodes = [node]
        print(node)
        successors = origGraph.successors(node) # get nodes which feed into this latch/output
        listOfAdj=[]

        # Add found nodes to searchlist for this node
        searchList = [node]
        # print(searchList)
        
        # Iterate through search list after addin entries
        while searchList:
            currentNode = searchList.pop(0)
            seenNodes += [currentNode]
            # print(f"Searching:{currentNode} Seen: {seenNodes}-------- To-Search{searchList}")
            successors = getDiGraphSuccessors(origGraph, currentNode)
            predecessors = getDiGraphPredecessors(origGraph, currentNode)
            adj = list(set(successors + predecessors))

            # Add nodes not seen before into list of future search nodes
            newNodes = [x for x in adj if (x not in seenNodes) and (x not in searchList) and not any(y in x for y in IOLList)]
            otherNodes = [x for x in adj if any(y in x for y in IOLList)]

            # When we find a terminating node, add it to the list of found terminating nodes and ignore them for the future
            OLFound += [x for x in otherNodes if any(y in x for y in OLList) and x[0] != "I"]
            
            # print(otherNodes)
            seenNodes = list(set(seenNodes + otherNodes))
            # print(seenNodes)
            # Get nodes who match IOLList
            predecessors = [x for x in predecessors]
            adj = [x for x in adj]
            # print(f"   Successors for {currentNode}: {newNodes}")
            # print(f"   Predecessors for {currentNode}: {predecessors}")
            # print(f"   Adjacent Nodes for {currentNode}: {adj}")
            # print(newNodes)
            searchList += newNodes


        # Create graph using connected nodes

        subGraph = origGraph.subgraph(seenNodes).copy()
        # print(origGraph.nodes())
        # print(seenNodes)
        print(subGraph.nodes())
        print(OLFound)
        # print(subGraph.edges())


        # Store found graph in graphList under name of starter node, and store current node in outputList
        outputList += [node]
        graphList += {subGraph}

pprint(graphList)
graph = graphList[0]
print(graph.order())
print([n for n in graph])
