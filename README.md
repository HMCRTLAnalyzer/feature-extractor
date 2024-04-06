TODO: populate readme

#### Feature Finder: Scripts to create, detect, and analyze graphs generated from RTL projects

### Graph generation

Yosys is used to generate graphs through graphviz for analysis. These graphs are dumped into the graphs directory based on the top module's name. Yosys performs basic optimizations when processing for the graph, such as removing unused gates. To generate graphs, create an RTL project and call `create_graphs.py` with the correct parameters. `-v [PROJECT SOURCE DIR]` points to the original project's source directory while `-dc [SYNTH RESULTS DIR]` points to the directory containing the synthesized netlist.

Finally, tell the script what the top module in the design is named in the design using `-T [MYTOPMODULE]`

At the moment, all graphs are produced using yosys and ABC combined to synthesize the designs as fast as possible. It is also possible to look at the assign-level statements.

# Example call
```./create_graphs.py -v "RTL/myprojectname/src" -dc "../results_dir/myprojectname/" -T "top"```

Graphs will be dumped under the directory `graphs/[MYTOPMODULE]/` in the `.dot` format. These can be visualized with a graphviz viewer and analyzed with any graphviz library of your choice.

# AIGER format
```./create_aig```

Graphs can also be made in the .AIGER format developed by Armin Biere. This uses the AIGER format as an intermediary before generating a .dot file to be used with the included python scripts. The `create_aig.py` script will run yosys synthesis, pass the circuit through ABC, and then map it to an and-inverter graph (AIG) with latches in the AIGER format.

This must be setup before running `./create_aig.py`. Run the script `setup.sh` or the command `./setup.sh` in this directory before running the `create_aig.py` script.

# TODO:
- Make create_aig.py actually call aigtodot within submodule
- Create list of inputs and outputs within AIGER format
- Figure out how to use yosys techmap to analyze output

### Feature Finding


#
