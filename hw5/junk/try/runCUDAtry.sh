#!/bin/bash

# This script is used to run the CUDA GPU code and get the time for both 5000x5000 and 10000x10000 grids. The outputs are stored in the scratch folder.

# echo "This script runs the CUDA code 3 times to get an average time taken for both 5000x5000 and 10000x10000 grids. It also does the same for the improved versions."
# echo "Another script runs the MPI code once for each grid size so that the output can be tested later."
# echo "The outputs will be stored in the scratch folder, and this script will compare the outputs of all the versions to ensure they are the same."
# echo "The CUDA code will be given 1 CPU and 1 GPU as well as 8gb of memory."
# echo "The MPI code will be given 1 CPU and 8gb of memory (This counts as the single threaded CPU version for comparison)."


source /apps/profiles/modules_asax.sh.dyn

# Load the CUDA module
module load cuda/11.7.0

#Compile programs
echo "Compiling CUDA programs..."
nvcc -o lifeCUDAtry lifeCUDAtry.cu
nvcc -o lifeCUDAimprovedtry lifeCUDAimprovedtry.cu

# Run the executable with the necessary arguments
echo "Running CUDA programs..."
echo "CUDA 5000x5000 3 tests"
time ./lifeCUDAtry 5000 5000 500 1 /home/$USER/hw5/try/outputs
echo "------------------------------------------------------------------------------------------------------------------------"
echo "CUDA 10000x10000 3 tests"
# time ./lifeCUDAtry 10000 10000 5000 1 /home/$USER/hw5/try/outputs
echo "------------------------------------------------------------------------------------------------------------------------"
echo "CUDA Improved 5000x5000 3 tests"
time ./lifeCUDAimprovedtry 5000 5000 500 1 /home/$USER/hw5/try/outputs
echo "------------------------------------------------------------------------------------------------------------------------"
echo "CUDA Improved 10000x10000 3 tests"
# time ./lifeCUDAimprovedtry 10000 10000 5000 1 /home/$USER/hw5/try/outputs
echo "------------------------------------------------------------------------------------------------------------------------"
# echo "All tests complete. Comparing outputs..."
# echo "Outputs compared:"
# echo "      5000x5000 CUDA vs MPI"
# echo "      10000x10000 CUDA vs MPI"
# echo "      5000x5000 CUDA vs CUDA Improved"
# echo "      10000x10000 CUDA vs CUDA Improved"
# echo ""
# echo ""
# echo ""
# echo "\/   \/   \/ "
# diff /scratch/$USER/cudaOutputs/output.5000.5000.1.cuda.txt /scratch/$USER/mpiOutputs/output.5000.5000.1.mpi.txt
# diff /scratch/$USER/cudaOutputs/output.10000.5000.1.cuda.txt /scratch/$USER/mpiOutputs/output.10000.5000.1.mpi.txt
# diff /scratch/$USER/cudaOutputs/output.5000.5000.1.cuda.txt /scratch/$USER/cudaImprovedOutputs/output.5000.5000.1.cudaimp.txt
# diff /scratch/$USER/cudaOutputs/output.10000.5000.1.cuda.txt /scratch/$USER/cudaImprovedOutputs/output.10000.5000.1.cudaimp.txt
# echo "/\   /\   /\ "
# echo "If no output was shown above, then the outputs are the same and the CUDA code is correct!"

