TODO: populate readme

#### Feature Finder: Smorgasborg of scripts to create, detect, and analyze graphs generated from RTL projects

### Graph generation

Yosys is used to generate graphs through graphviz for analysis. These graphs are dumped into the graphs directory based on the top module's name. Yosys performs basic optimizations when processing for the graph, such as removing unused gates. To generate graphs, create an RTL project and call `create_graphs.py` with the correct parameters. `-v [PROJECT SOURCE DIR]` points to the original project's source directory while `-dc [SYNTH RESULTS DIR]` points to the directory containing the synthesized netlist.

Finally, tell the script what the top module in the design is named in the design using `-T [MYTOPMODULE]`

# Example call
```./create_graphs.py -v "RTL/myprojectname/src" -dc "../results_dir/myprojectname/" -T "top"```

Graphs will be dumped under the directory `graphs/[MYTOPMODULE]/` in the `.dot` format. These can be visualized with a graphviz viewer and analyzed with any graphviz library of your choice.

### Feature Finding


#
