# ---------------- CONFIG -----------------------
export SKY130_PATH := /opt/riscv/cad/lib/sky130_osu_sc_t12/12T_hs/lib
# HDL_LANG option is either verilog or sverilog. default to verilog unless set otherwise
export HDL_LANG ?= verilog

time := $(shell date +%F-%H-%M-%S)

# IMPORTANT: MODULE_NAME must be the name of the TOP module in the design! 
ifndef MODULE_NAME
$(error MODULE_NAME not set, please set in order to continue)
endif
export MODULE_NAME

ifndef SRC_PATH
$(error SRC_PATH not set, please set in order to continue)
endif
export SRC_PATH

ifndef NAME
GRAPH_DIR = ./graphs/$(NAME)
else
GRAPH_DIR = ./graphs/$(NAME)
endif
export GRAPH_DIR

build:
	echo Building aiger library
	cd aiger && ./configure.sh && make

graph: 
	mkdir $(GRAPH_DIR) && \
	yosys -c graph_gen.tcl | tee $(GRAPH_DIR) graph_gen.log

aig:
	mkdir $(GRAPH_DIR) && \
	yosys -c aig_gen.tcl | tee $(GRAPH_DIR)/aig_gen.log

# generate dot from aiger using aigtodot utility
	./aigtodot $(GRAPH_DIR)/$(MODULE_NAME).aig $(GRAPH_DIR)/$(MODULE_NAME).dot

aig_auto:
	yosys -c aig_gen.tcl | tee $(GRAPH_DIR)/aig_gen.log

# generate dot from aiger using aigtodot utility
	./aigtodot $(GRAPH_DIR)/$(MODULE_NAME).aig $(GRAPH_DIR)/$(MODULE_NAME).dot


clean_work:
	@if [ -d WORK ]; then \
        echo "Deleting contents of WORK folder"; \
        rm -rf WORK/*; \
    fi

clean_graph:
	rm -rf graphs/*

clear_aiger:
	echo Removing aiger library
	cd aiger & make clean
