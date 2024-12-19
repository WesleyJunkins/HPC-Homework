Wesley Junkins, CS 581

Homework 5: A GPU implementation of the Game of Life

Repository Structure:
- IO FILES: Contains the input and output files from running the CUDA script.
- SCRIPTS: Contains the shell scripts used to run the CUDA and MPI code.
    - runCUDA.sh: Runs all versions of the CUDA program (5000x5000, 10000x10000, and improved version), as well as uses diff to compare the outputs.
    - runMPI.sh: Runs the MPI code on a single processor without multithreading to act as the serial CPU program used to test the output from the CUDA versions. This MPI program was verified to work correctly in previous homeworks.
    - runMPItime.sh: Runs the MPI code again with 20 processor cores to get the time taken.
- SOURCE CODE: Contains the source code for the CUDA and MPI versions. How to compile and run each program is specified in the header of each file.
    - lifeCUDA.cu: The original CUDA implementation without improvements or using shared memory.
    - lifeCUDAimproved.cu: The improved CUDA implementation using shared memory and a 16x16 thread grid per block for a performance boost as claimed in the paper on Blackboard.
    - lifeMPI.cpp: The MPI version that was used as the serial code to test the output of the GPU code.
- junk: Contains files that I didn't need anymore but was too scared to delete.