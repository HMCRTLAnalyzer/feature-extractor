#!/usr/bin/python3
#   -----------------------------------------   #
#   Script to generate dot files from general-synth results   
#   Written by Kaitlin Lucio 11/27/23
#   
#   CURRENTLY BROKEN DUE TO CHANGES MADE FOR PYTHON TO FUNCTION
#   -----------------------------------------   #

# Import libraries

from pprint import *
import argparse
import os

# Code for argparse

parser = argparse.ArgumentParser(
                    prog='create_graphs',
                    description='Create .dot graph representations of verilog files using yosys!',
                    epilog='For more info contact nlucio@hmc.edu')

parser.add_argument('--verilog_dir', '-v', dest='verilog_dir', default="", type=str, required=True,
                    help='filepath to a source directory for a project')

parser.add_argument('--synth_verilog', '-dc', dest='synth_verilog', default="", type=str, required=False,
                    help='filepath to the synthesized verilog')

parser.add_argument('-T', '--TOPMODULE', dest='topmodule', default="", type=str, required=True,
                        help='name of top module')

parser.add_argument('-L', '--LANG', dest='language', default="verilog", type=str, required=False,
                        help='Language of project')

args = parser.parse_args()

def main():
    top = args.topmodule
    results_path = f"./graphs/{top}"
    lang = args.language
    orig_src = args.verilog_dir
    synth_src = args.synth_verilog

    if not os.path.exists(results_path):
        os.mkdir(results_path)

    # Call makefile for original project's verilog
    os.system(f"make aig NAME={top} HDL_LANG={lang} MODULE_NAME={top} GRAPH_NAME={top} SRC_PATH={orig_src}")

    # If synth_verilog is defined, call makefile on that as well and put results in same directory as above
    if synth_src != "":
        os.system(f"make aig NAME={top} HDL_LANG={lang} MODULE_NAME={top} GRAPH_NAME={top}_SYNTH_AIG SRC_PATH={synth_src}")



if __name__ == "__main__":
    main()
