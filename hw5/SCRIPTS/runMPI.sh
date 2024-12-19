#!/bin/bash

# This is the script used to get the output from the serial MPI program. This program was verified previously to have correct output. It will be used to test the output of the CUDA GPU code.

source /apps/profiles/modules_asax.sh.dyn
module load openmpi/4.1.4-gcc11
mpic++ lifeMPI.cpp -O -o lifeMPI -std=c++11

echo "MPI 5000x5000 1 test"
mpirun -n 1 ./lifeMPI 5000 5000 5000 1 /scratch/$USER/mpiOutputs
echo "MPI 10000x10000 1 test"
mpirun -n 1 ./lifeMPI 10000 10000 5000 1 /scratch/$USER/mpiOutputs