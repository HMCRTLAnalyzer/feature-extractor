# TODO: figure out how to use liberty files to get graph from DC netlist if library.tcl not functional
source ./library.tcl

source ./set_src.tcl

set GRAPH_DIR $::env(GRAPH_DIR)
set MODULE_NAME $::env(MODULE_NAME)

# Process verilog files
yosys proc
yosys flatten
yosys memory
yosys techmap
yosys memory
yosys clean

# Synthesize and generate AIG
yosys abc -fast
yosys async2sync
yosys dffunmap
yosys aigmap
yosys stat
yosys write_aiger -zinit $GRAPH_DIR/$MODULE_NAME.aig

