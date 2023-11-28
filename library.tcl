##############################################################################
#                                                                            #
#                            SPECIFY LIBRARIES                               #
#                                                                            #
##############################################################################

set SKY130_PATH $::env(SKY130_PATH)

# Define worst case library
set LIB_WC_FILE   $SKY130_PATH/sky130_osu_sc_12T_hs_tt_1P20_25C.ccs.lib

# Define best case library
set LIB_BC_FILE   $SKY130_PATH/sky130_osu_sc_12T_hs_tt_1P20_25C.ccs.lib

# Set library
set target_library $LIB_WC_FILE
set link_library   $LIB_WC_FILE

# Read worst case liberty files
yosys read_liberty $LIB_WC_FILE

# # Read best case liberty files
# yosys read_liberty $LIB_BC_FILE

