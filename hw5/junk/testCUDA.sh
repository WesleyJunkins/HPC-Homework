#!/bin/bash

source /apps/profiles/modules_asax.sh.dyn

# Load the CUDA module
module load cuda/11.7.0
module load openmpi/4.1.4-gcc11


#Compile programs
echo "Compiling CUDA programs..."
nvcc -o lifeCUDAtry lifeCUDAtry.cu
# nvcc -o lifeCUDAimproved lifeCUDAimproved.cu
# mpic++ lifeMPI.cpp -O -o lifeMPI -std=c++11


# Run the executable with the necessary arguments
echo "Running CUDA programs..."
./lifeCUDAtry 10000 10000 100 1 /home/$USER/hw5/testOutputs
# ./lifeCUDAimproved 100 100 20 1 /home/$USER/hw5/testOutputs
# mpirun -n 1 ./lifeMPI 100 100 20 20 /home/$USER/hw5/testOutputs

# diff /home/$USER/hw5/testOutputs/output.100.20.1.cuda.txt /home/$USER/hw5/testOutputs/output.100.20.20.mpi.txt
# diff /home/$USER/hw5/testOutputs/output.100.20.1.cuda.txt /home/$USER/hw5/testOutputs/output.100.20.1.cudaimp.txt




