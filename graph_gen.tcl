# TODO: figure out how to use liberty files to get graph from DC netlist if library.tcl not functional
source ./library.tcl

source ./set_src.tcl

set GRAPH_DIR $::env(GRAPH_DIR)
set GRAPH_NAME $::env(GRAPH_NAME)

# Process verilog files
yosys proc
yosys flatten
yosys memory
yosys techmap
yosys memory
yosys clean

# output graphviz files to specified graph_dir
yosys show -colors 1 -format dot -width -stretch -prefix $GRAPH_DIR/$GRAPH_NAME
