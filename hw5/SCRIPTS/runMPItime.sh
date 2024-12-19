#!/bin/bash

# Gets the 20 core time taken by MPI program
source /apps/profiles/modules_asax.sh.dyn
module load openmpi/4.1.4-gcc11
mpic++ lifeMPI.cpp -O -o lifeMPI -std=c++11

echo "MPI 5000x5000 1 test"
time mpirun -n 20 ./lifeMPI 5000 5000 5000 20 /scratch/$USER
echo "MPI 10000x10000 1 test"
time mpirun -n 20 ./lifeMPI 10000 10000 5000 20 /scratch/$USER