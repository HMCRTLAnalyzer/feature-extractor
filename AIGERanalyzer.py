#!/bin/python3

# Import libraries
from aigerdotlib import *
import numpy as np
import pandas as pd
import pickle

csv_filepath = "diffs_test_paths.csv"
pickle_dirpath = "./graph_pickles"
regenerate_pickles = 0
openABCD_df = pd.read_csv(csv_filepath)
openABCD_df = openABCD_df.loc[:,["module","path_to_rtl","language"]]
print(openABCD_df)

IO_basenames = ["I","O","L"]
output_basenames = ["O","L"]

stats_df = pd.DataFrame()

graph_dict = {}
graph_dict["module"] = []

for row in openABCD_df.itertuples(index=False):
    module = row[0]
    path_to_rtl = row[1]
    language = row[2]
    graph_filepath = f"{pickle_dirpath}/{module}_graphs.pickle"
    input_filepath = f"{pickle_dirpath}/{module}_inputs.pickle"
    output_filepath = f"{pickle_dirpath}/{module}_outputs.pickle"
    
    # Check if pickle exists. if it does, load the pickle instead of regenerating graph list.
    try:
        if regenerate_pickles == 1:
            dot_filepath = generateDOT(path_to_rtl, module, language)
            print(f"Generating pickle for {module}")
            [graph_list, output_names, input_names] = cutAIGERtoDAGs(dot_filepath, IO_basenames, output_basenames)
            for graph in graph_list:
                NotEdgetoNotNode(graph)
            with open(graph_filepath, "wb") as handle:
                pickle.dump(graph_list, handle)
            with open(input_filepath, "wb") as handle:
                pickle.dump(input_names, handle)
            with open(output_filepath, "wb") as handle:
                pickle.dump(output_names, handle)

        graph_file = open(graph_filepath,"rb")
        graph_list = pickle.load(graph_file)
        input_file = open(input_filepath,"rb")
        input_names = pickle.load(input_file)
        output_file = open(output_filepath,"rb")
        output_names = pickle.load(output_file)
    except:
        dot_filepath = generateDOT(path_to_rtl, module, language)
        print(f"Generating pickle for {module}")
        [graph_list, output_names, input_names] = cutAIGERtoDAGs(dot_filepath, IO_basenames, output_basenames)
        for graph in graph_list:
            NotEdgetoNotNode(graph)
        with open(graph_filepath, "wb") as handle:
            pickle.dump(graph_list, handle)
        with open(input_filepath, "wb") as handle:
            pickle.dump(input_names, handle)
        with open(output_filepath, "wb") as handle:
            pickle.dump(output_names, handle)

    graph_dict["module"] = graph_list

    print("PYTHON OUTPUT")
    print()
    print([n for n in graph_list[0].nodes()])


