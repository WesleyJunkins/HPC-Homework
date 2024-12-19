#!/bin/bash

# This script is used to compare the outputs of the CUDA code with the MPI code and the improved CUDA code.

echo "Outputs compared:"
echo "      5000x5000 CUDA vs MPI"
echo "      10000x10000 CUDA vs MPI"
echo "      5000x5000 CUDA vs CUDA Improved"
echo "      10000x10000 CUDA vs CUDA Improved"
echo ""
echo ""
echo ""
echo "\/   \/   \/ "
echo "1. 5000x5000 CUDA vs MPI"
diff /scratch/$USER/cudaOutputs/output.5000.5000.1.cuda.txt /scratch/$USER/mpiOutputs/output.5000.5000.1.mpi.txt
echo "2. 10000x10000 CUDA vs MPI"
diff /scratch/$USER/cudaOutputs/output.10000.5000.1.cuda.txt /scratch/$USER/mpiOutputs/output.10000.5000.1.mpi.txt
echo "3. 5000x5000 CUDA vs CUDA Improved"
diff /scratch/$USER/cudaOutputs/output.5000.5000.1.cuda.txt /scratch/$USER/cudaImprovedOutputs/output.5000.5000.1.cudaimp.txt
echo "4. 10000x10000 CUDA vs CUDA Improved"
diff /scratch/$USER/cudaOutputs/output.10000.5000.1.cuda.txt /scratch/$USER/cudaImprovedOutputs/output.10000.5000.1.cudaimp.txt
echo "/\   /\   /\ "
echo "If no extra output was shown above, then the outputs are the same and the CUDA code is correct!"