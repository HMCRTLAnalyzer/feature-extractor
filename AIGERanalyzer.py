#!/bin/python3

# Import libraries
from aigerdotlib import *
import numpy as np
import pandas as pd
import pickle
import argparse





csv_in_filepath = "diffs_test_paths.csv"
csv_out_filepath = "teststats.csv"
pickle_dirpath = "./graph_pickles"
regenerate_pickles = 0
openABCD_df = pd.read_csv(csv_in_filepath)
openABCD_df = openABCD_df.loc[:,["module","path_to_rtl","language","sensitive","memory"]]
print(openABCD_df)

# Define Parameters for dataframe and analysis
IO_basenames = ["I","O","L"]
output_basenames = ["O","L"]
N = 5

nlist = list(range(N))
nlist.sort(reverse=True)
ld_cols = [f"LD{x}" for x in nlist]
le_cols = [f"LE{x}" for x in nlist]
nodecnt_cols = [f"NCNT{x}" for x in nlist]
fanout_cols = [f"FO{x}" for x in nlist]

stats_col = ["module","sensitive","memory"]+ld_cols+le_cols+nodecnt_cols+fanout_cols
df_list = []

graph_dict = {}
graph_dict["module"] = []

for row in openABCD_df.itertuples(index=False):
    module = row[0]
    path_to_rtl = row[1]
    language = row[2]
    sensitive = row[3]
    memory = row[4]
    graph_filepath = f"{pickle_dirpath}/{module}_graphs.pickle"
    
    # Check if pickle exists. if it does, load the pickle instead of regenerating graph list.
    try:
        if regenerate_pickles == 1:
            dot_filepath = generateDOT(path_to_rtl, module, language)
            print(f"Generating pickle for {module}")
            graph_dict = cutAIGERtoDAGs(dot_filepath, IO_basenames, output_basenames)
            for key, entry in graph_dict.items():
                graph = entry["graph"]
                replaceEdgesWithNotNodes(graph)
                replaceBufWithLatch(graph)
                logicalEffortEdgeMap(graph)
            with open(graph_filepath, "wb") as handle:
                pickle.dump(graph_dict, handle)
        
        graph_file = open(graph_filepath,"rb")
        graph_dict = pickle.load(graph_file)
    except:
        dot_filepath = generateDOT(path_to_rtl, module, language)
        print(f"Generating pickle for {module}")
        graph_dict = cutAIGERtoDAGs(dot_filepath, IO_basenames, output_basenames)
        idx = 0
        for key, entry in graph_dict.items():
            graph = entry["graph"]
            replaceEdgesWithNotNodes(graph)
            replaceBufWithLatch(graph)
            logicalEffortEdgeMap(graph)
            nx.nx_pydot.write_dot(graph,f"{idx}.dot")
            idx += 1
        with open(graph_filepath, "wb") as handle:
            pickle.dump(graph_dict, handle)

    LD = topNLargestLD(graph_dict,N)
    LE = topNLargestLE(graph_dict,N)
    LNC = topNCLBNodeCount(graph_dict,N)
    FN = topNFanouts(graph_dict,N)
    vals = [module, sensitive, memory]+LD+LE+LNC+FN

    df_list += [dict(zip(stats_col,vals))]

# After getting all df items, create dataframe and save it to a csv.
stats_df = pd.DataFrame(data=df_list)
stats_df.to_csv(csv_out_filepath)
