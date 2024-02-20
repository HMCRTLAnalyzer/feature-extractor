set src_dir $::env(SRC_PATH)
set MODULE_NAME $::env(MODULE_NAME)
set GRAPH_DIR $::env(GRAPH_DIR)

# check for files with the correct file extension
if {$::env(HDL_LANG) == "verilog"} {
    set file_ext "*.v"
    set config_ext "*.vh"
} elseif {$::env(HDL_LANG) == "sverilog"} {
    set file_ext "*.sv"
    set config_ext "*.vh"
} else {
    set file_ext "*.vhdl"
    # VHDL does not use headers
    set config_ext "" 
}

# Commented to avoid recursive issues when trying to define top level module from results.
# # findFiles, from https://stackoverflow.com/questions/429386/tcl-recursively-search-subdirectories-to-source-all-tcl-files
# # basedir - the directory to start looking in
# # pattern - A pattern, as defined by the glob command, that the files must match
# proc findFiles { basedir pattern } {

#     # Fix the directory name, this ensures the directory name is in the
#     # native format for the platform and contains a final directory seperator
#     set basedir [string trimright [file join [file normalize $basedir] { }]]
#     set fileList {}

#     # Look in the current directory for matching files, -type {f r}
#     # means ony readable normal files are looked at, -nocomplain stops
#     # an error being thrown if the returned list is empty

# foreach fileName [glob -nocomplain -type {f r} -path $basedir $pattern] {
#         lappend fileList $fileName
#     }

#     # Now look for any sub direcories in the current directory
#     foreach dirName [glob -nocomplain -type {d  r} -path $basedir *] {
#         # Recusively call the routine on the sub directory and append any
#         # new files to the results
#         set subDirList [findFiles $dirName $pattern]
#         if { [llength $subDirList] > 0 } {
#             foreach subDirFile $subDirList {
#                 lappend fileList $subDirFile
#             }
#         }
#     }
#     return $fileList
#  }


# append config and source files
set src_files $src_dir/$file_ext

# Read files into yosys
yosys read_verilog -sv $src_files

# process imported verilog files
yosys hierarchy -check -top $MODULE_NAME
