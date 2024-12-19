#!/bin/bash

# To compile: icpx -qopenmp -O -o lifeOMPicpx lifeOMP.cpp

module load intel

# Running each round once
echo "ICPX Compiled"
echo "Round 1"
echo "--------------------------------"
echo "1 Thread"
time ./lifeOMPicpx 5000 5000 1 5000 /scratch/$USER
echo "2 Threads"
time ./lifeOMPicpx 5000 5000 2 5000 /scratch/$USER
echo "4 Threads"
time ./lifeOMPicpx 5000 5000 4 5000 /scratch/$USER
echo "8 Threads"
time ./lifeOMPicpx 5000 5000 8 5000 /scratch/$USER
echo "10 Threads"
time ./lifeOMPicpx 5000 5000 10 5000 /scratch/$USER
echo "16 Threads"
time ./lifeOMPicpx 5000 5000 16 5000 /scratch/$USER
echo "20 Threads"
time ./lifeOMPicpx 5000 5000 20 5000 /scratch/$USER
echo "--------------------------------"
echo " "

# Running each round again, we do not run them together to avoid any possible interference
echo "Round 2"
echo "--------------------------------"
echo "1 Thread"
time ./lifeOMPicpx 5000 5000 1 5000 /scratch/$USER
echo "2 Threads"
time ./lifeOMPicpx 5000 5000 2 5000 /scratch/$USER
echo "4 Threads"
time ./lifeOMPicpx 5000 5000 4 5000 /scratch/$USER
echo "8 Threads"
time ./lifeOMPicpx 5000 5000 8 5000 /scratch/$USER
echo "10 Threads"
time ./lifeOMPicpx 5000 5000 10 5000 /scratch/$USER
echo "16 Threads"
time ./lifeOMPicpx 5000 5000 16 5000 /scratch/$USER
echo "20 Threads"
time ./lifeOMPicpx 5000 5000 20 5000 /scratch/$USER
echo "--------------------------------"
echo " "

# Run a third time
echo "Round 3"
echo "--------------------------------"
echo "1 Thread"
time ./lifeOMPicpx 5000 5000 1 5000 /scratch/$USER
echo "2 Threads"
time ./lifeOMPicpx 5000 5000 2 5000 /scratch/$USER
echo "4 Threads"
time ./lifeOMPicpx 5000 5000 4 5000 /scratch/$USER
echo "8 Threads"
time ./lifeOMPicpx 5000 5000 8 5000 /scratch/$USER
echo "10 Threads"
time ./lifeOMPicpx 5000 5000 10 5000 /scratch/$USER
echo "16 Threads"
time ./lifeOMPicpx 5000 5000 16 5000 /scratch/$USER
echo "20 Threads"
time ./lifeOMPicpx 5000 5000 20 5000 /scratch/$USER
echo "--------------------------------"
echo " "