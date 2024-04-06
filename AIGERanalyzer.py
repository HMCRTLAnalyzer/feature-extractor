#!/bin/python3

# Import libraries
from aigerdotlib import *
import os
import numpy as np
import pandas as pd
import pickle
import argparse
import logging
import time

N = 350
# csv_in_filepath = "/home/qualcomm_clinic/RTL_dataset/training_data.csv"
# csv_out_filepath = f"processed_data/top{N}_MLdata.csv"
csv_temp_filepath = f"processed_data/temp_rob{N}.csv"
csv_in_filepath = "bruh2.csv"
csv_out_filepath = "Robstats.csv"
curr_time = time.strftime("%Y-%d_%H%M")
logfile = f"logs/run_{curr_time}.log"
pickle_dirpath = "./graph_pickles"
regenerate_pickles = 0
openABCD_df = pd.read_csv(csv_in_filepath)
openABCD_df = openABCD_df.loc[:,["module","path_to_rtl","language","sensitive","memory"]]
print(openABCD_df)

# Define Parameters for dataframe and analysis
IO_basenames = ["I","O","L"]
output_basenames = ["O","L"]

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

# Write stats_col to csv file. ASSUMES YOU ARE ON LINUX
data_out = ",".join(map(str, stats_col))
os.system(f"echo {data_out} > {csv_temp_filepath}")

logger = logging.getLogger(__name__)
logging.basicConfig(filename=logfile,level=logging.DEBUG)

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
            idx = 0
            for key, entry in graph_dict.items():
                graph = entry["graph"]
                replaceEdgesWithNotNodes(graph)
                replaceBufWithLatch(graph)
                logicalEffortEdgeMap(graph)
                idx += 1
            with open(graph_filepath, "wb") as handle:
                pickle.dump(graph_dict, handle)
        
        graph_file = open(graph_filepath,"rb")
        graph_dict = pickle.load(graph_file)
    except Exception as e:
        try:
            dot_filepath = generateDOT(path_to_rtl, module, language)
            print(f"Generating pickle for {module}")
            graph_dict = cutAIGERtoDAGs(dot_filepath, IO_basenames, output_basenames)
            idx = 0
            for key, entry in graph_dict.items():
                graph = entry["graph"]
                replaceEdgesWithNotNodes(graph)
                replaceBufWithLatch(graph)
                logicalEffortEdgeMap(graph)
            with open(graph_filepath, "wb") as handle:
                pickle.dump(graph_dict, handle)
        except Exception as e:
            logger.error(f"Synthesis failed for {module} with error {e}")
    finally:
        pass
    logger.debug(graph_dict)
    logger.info(f"Analyzing {module}")
    LD = topNLargestLD(graph_dict,N)
    LE = topNLargestLE(graph_dict,N)
    LNC = topNCLBNodeCount(graph_dict,N)
    FN = topNFanouts(graph_dict,N)
    vals = [module, sensitive, memory]+LD+LE+LNC+FN

    logger.info(f"Writing saved stats to existing temp csv file")
    data_line = [module]+[sensitive]+[memory]+LD+LE+LNC+FN
    data_out = ",".join(map(str, [str(x) for x in data_line]))
    os.system(f"echo {data_line} >> {csv_temp_filepath}")

    df_list += [dict(zip(stats_col,vals))]

# After getting all df items, create dataframe and save it to a csv.
logger.info(f"Wrote dataset csv to {csv_out_filepath}")
stats_df = pd.DataFrame(data=df_list)
stats_df.to_csv(csv_out_filepath)
