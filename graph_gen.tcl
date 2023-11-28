# TODO: figure out how to use liberty files to get graph from DC netlist if library.tcl not functional
source ./library.tcl

source ./set_src.tcl

set GRAPH_DIR $::env(GRAPH_DIR)
set GRAPH_NAME $::env(GRAPH_NAME)

# output graphviz files to specified graph_dir
yosys show -colors 1 -format dot -width -stretch -prefix $GRAPH_DIR/$GRAPH_NAME

# Include png file for human legibility (TODO: REMOVE ONCE TESTING IS DONE)
yosys show -colors 1 -format png -width -stretch -prefix $GRAPH_DIR/$GRAPH_NAME
