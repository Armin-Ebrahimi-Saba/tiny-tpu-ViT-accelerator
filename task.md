- we are using Xilinx Artix-7 XC7A200T
- my final goal to run the inference of the depth-anything v2 on this board
- I want to use the smallest version of this model
- model weights should be stored on the DRAM (DDR3)
- reason about each step and document the steps that you took in a report.md file
- you should write in the student-related files
- tiny-tpu should eventually be integrated in rvlab project
- first read the docs/ to get yourself familiar with the structure of the program
- you may ask me questions at the beginning if anything is not clear

### after running bitstream, pnr and syn check the following files for warnings and fix them without changing third party libraries if possible

- build/rvlab_fpga_top/bitstream/rvlab_fpga_top.io_report.txt
- build/rvlab_fpga_top/syn/rvlab_fpga_top.*.txt
- build/rvlab_fpga_top/pnr/rvlab_fpga_top.*.txt