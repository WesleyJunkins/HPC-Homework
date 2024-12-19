#!/bin/bash
source /apps/profiles/modules_asax.sh.dyn
module load openmpi/4.1.4-gcc11

echo "Starting MPI Tests"

echo ""

echo "Compiling"
mpic++ lifeMPI.cpp -O -o lifeMPI -std=c++11
mpic++ lifeMPInoblock.cpp -O -o lifeMPInoblock -std=c++11

echo ""

echo "Running each test 3 times (first, the blocking code)"
echo ""

echo "--------------------"
echo "1 Processor"
time mpirun -n 1 ./lifeMPI 5000 5000 5000 1 /scratch/$USER
time mpirun -n 1 ./lifeMPI 5000 5000 5000 1 /scratch/$USER
time mpirun -n 1 ./lifeMPI 5000 5000 5000 1 /scratch/$USER
echo "--------------------"

echo ""

echo "--------------------"
echo "2 Processos"
time mpirun -n 2 ./lifeMPI 5000 5000 5000 2 /scratch/$USER
time mpirun -n 2 ./lifeMPI 5000 5000 5000 2 /scratch/$USER
time mpirun -n 2 ./lifeMPI 5000 5000 5000 2 /scratch/$USER
echo "--------------------"

echo ""

echo "--------------------"
echo "4 Processos"
time mpirun -n 4 ./lifeMPI 5000 5000 5000 4 /scratch/$USER
time mpirun -n 4 ./lifeMPI 5000 5000 5000 4 /scratch/$USER
time mpirun -n 4 ./lifeMPI 5000 5000 5000 4 /scratch/$USER
echo "--------------------"

echo ""

echo "--------------------"
echo "8 Processos"
time mpirun -n 8 ./lifeMPI 5000 5000 5000 8 /scratch/$USER
time mpirun -n 8 ./lifeMPI 5000 5000 5000 8 /scratch/$USER
time mpirun -n 8 ./lifeMPI 5000 5000 5000 8 /scratch/$USER
echo "--------------------"

echo ""

echo "--------------------"
echo "10 Processos"
time mpirun -n 10 ./lifeMPI 5000 5000 5000 10 /scratch/$USER
time mpirun -n 10 ./lifeMPI 5000 5000 5000 10 /scratch/$USER
time mpirun -n 10 ./lifeMPI 5000 5000 5000 10 /scratch/$USER
echo "--------------------"

echo ""

echo "--------------------"
echo "16 Processos"
time mpirun -n 16 ./lifeMPI 5000 5000 5000 16 /scratch/$USER
time mpirun -n 16 ./lifeMPI 5000 5000 5000 16 /scratch/$USER
time mpirun -n 16 ./lifeMPI 5000 5000 5000 16 /scratch/$USER
echo "--------------------"

echo ""

echo "--------------------"
echo "20 Processos"
time mpirun -n 20 ./lifeMPI 5000 5000 5000 20 /scratch/$USER
time mpirun -n 20 ./lifeMPI 5000 5000 5000 20 /scratch/$USER
time mpirun -n 20 ./lifeMPI 5000 5000 5000 20 /scratch/$USER
echo "--------------------"

echo ""
echo ""
echo "Running the non-blocking code"
echo ""
echo ""



echo "--------------------"
echo "1 Processor"
time mpirun -n 1 ./lifeMPInoblock 5000 5000 5000 1 /scratch/$USER/noblockres
time mpirun -n 1 ./lifeMPInoblock 5000 5000 5000 1 /scratch/$USER/noblockres
time mpirun -n 1 ./lifeMPInoblock 5000 5000 5000 1 /scratch/$USER/noblockres
echo "--------------------"

echo ""

echo "--------------------"
echo "2 Processos"
time mpirun -n 2 ./lifeMPInoblock 5000 5000 5000 2 /scratch/$USER/noblockres
time mpirun -n 2 ./lifeMPInoblock 5000 5000 5000 2 /scratch/$USER/noblockres
time mpirun -n 2 ./lifeMPInoblock 5000 5000 5000 2 /scratch/$USER/noblockres
echo "--------------------"

echo ""

echo "--------------------"
echo "4 Processos"
time mpirun -n 4 ./lifeMPInoblock 5000 5000 5000 4 /scratch/$USER/noblockres
time mpirun -n 4 ./lifeMPInoblock 5000 5000 5000 4 /scratch/$USER/noblockres
time mpirun -n 4 ./lifeMPInoblock 5000 5000 5000 4 /scratch/$USER/noblockres
echo "--------------------"

echo ""

echo "--------------------"
echo "8 Processos"
time mpirun -n 8 ./lifeMPInoblock 5000 5000 5000 8 /scratch/$USER/noblockres
time mpirun -n 8 ./lifeMPInoblock 5000 5000 5000 8 /scratch/$USER/noblockres
time mpirun -n 8 ./lifeMPInoblock 5000 5000 5000 8 /scratch/$USER/noblockres
echo "--------------------"

echo ""

echo "--------------------"
echo "10 Processos"
time mpirun -n 10 ./lifeMPInoblock 5000 5000 5000 10 /scratch/$USER/noblockres
time mpirun -n 10 ./lifeMPInoblock 5000 5000 5000 10 /scratch/$USER/noblockres
time mpirun -n 10 ./lifeMPInoblock 5000 5000 5000 10 /scratch/$USER/noblockres
echo "--------------------"

echo ""

echo "--------------------"
echo "16 Processos"
time mpirun -n 16 ./lifeMPInoblock 5000 5000 5000 16 /scratch/$USER/noblockres
time mpirun -n 16 ./lifeMPInoblock 5000 5000 5000 16 /scratch/$USER/noblockres
time mpirun -n 16 ./lifeMPInoblock 5000 5000 5000 16 /scratch/$USER/noblockres
echo "--------------------"

echo ""

echo "--------------------"
echo "20 Processos"
time mpirun -n 20 ./lifeMPInoblock 5000 5000 5000 20 /scratch/$USER/noblockres
time mpirun -n 20 ./lifeMPInoblock 5000 5000 5000 20 /scratch/$USER/noblockres
time mpirun -n 20 ./lifeMPInoblock 5000 5000 5000 20 /scratch/$USER/noblockres
echo "--------------------"

echo ""

echo "Finished MPI Tests"

