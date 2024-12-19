// By Wesley Junkins
// wcjunkins@crimson.ua.edu
//
// Homework 5: CUDA Game of Life (CS 581)
// This program implements the Game of Life using CUDA with shared memory. This is also the version that includes an improvement from the paper (using 16x16 thread grids per block. The paper claims this gives a performance boost).
//
// How to compile: nvcc lifeCUDAimproved.cu -o lifeCUDAimproved
// How to run: ./lifeCUDAimproved <numRows> <numCols> <maxGens> <numProcesses> <fileDirectory>

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <cuda.h>

using namespace std;

// Cell class remains unchanged
class cell
{
public:
    bool isGhost;
    int status;

    __host__ __device__ cell()
    {
        isGhost = false;
        status = 0;
    }

    __host__ __device__ void setAlive()
    {
        if (!isGhost)
            status = 1;
    }

    __host__ __device__ void setDead()
    {
        status = 0;
    }

    __host__ __device__ int getStatus() const
    {
        return status;
    }

    __host__ __device__ void setGhost()
    {
        isGhost = true;
        status = 0;
    }

    __host__ __device__ bool getGhostStatus() const
    {
        return isGhost;
    }
};

// Optimized Kernel for multiple generations
__global__ void processGenerations(cell *matrix, cell *newMatrix, int numRows, int numCols, int maxGens)
{
    extern __shared__ cell sharedMemory[];

    int globalRow = blockIdx.y * blockDim.y + threadIdx.y;
    int globalCol = blockIdx.x * blockDim.x + threadIdx.x;

    int sharedRow = threadIdx.y + 1; // Offset for shared memory to avoid boundary issues
    int sharedCol = threadIdx.x + 1; // Offset for shared memory

    int sharedWidth = blockDim.x + 2; // Add padding to accommodate ghost cells

    for (int gen = 0; gen < maxGens; gen++)
    {
        // Load cells into shared memory
        if (globalRow < numRows && globalCol < numCols)
        {
            sharedMemory[sharedRow * sharedWidth + sharedCol] = matrix[globalRow * numCols + globalCol];
        }

        // Load ghost cells for the boundaries
        if (threadIdx.y == 0 && globalRow > 0)
            sharedMemory[(sharedRow - 1) * sharedWidth + sharedCol] = matrix[(globalRow - 1) * numCols + globalCol];
        if (threadIdx.y == blockDim.y - 1 && globalRow < numRows - 1)
            sharedMemory[(sharedRow + 1) * sharedWidth + sharedCol] = matrix[(globalRow + 1) * numCols + globalCol];
        if (threadIdx.x == 0 && globalCol > 0)
            sharedMemory[sharedRow * sharedWidth + (sharedCol - 1)] = matrix[globalRow * numCols + (globalCol - 1)];
        if (threadIdx.x == blockDim.x - 1 && globalCol < numCols - 1)
            sharedMemory[sharedRow * sharedWidth + (sharedCol + 1)] = matrix[(globalRow)*numCols + (globalCol + 1)];

        // Load diagonal ghost cells
        if (threadIdx.y == 0 && threadIdx.x == 0 && globalRow > 0 && globalCol > 0)
            sharedMemory[(sharedRow - 1) * sharedWidth + (sharedCol - 1)] = matrix[(globalRow - 1) * numCols + (globalCol - 1)];
        if (threadIdx.y == 0 && threadIdx.x == blockDim.x - 1 && globalRow > 0 && globalCol < numCols - 1)
            sharedMemory[(sharedRow - 1) * sharedWidth + (sharedCol + 1)] = matrix[(globalRow - 1) * numCols + (globalCol + 1)];
        if (threadIdx.y == blockDim.y - 1 && threadIdx.x == 0 && globalRow < numRows - 1 && globalCol > 0)
            sharedMemory[(sharedRow + 1) * sharedWidth + (sharedCol - 1)] = matrix[(globalRow + 1) * numCols + (globalCol - 1)];
        if (threadIdx.y == blockDim.y - 1 && threadIdx.x == blockDim.x - 1 && globalRow < numRows - 1 && globalCol < numCols - 1)
            sharedMemory[(sharedRow + 1) * sharedWidth + (sharedCol + 1)] = matrix[(globalRow + 1) * numCols + (globalCol + 1)];

        // Synchronize to ensure all threads have completed boundary loading
        __syncthreads();

        // Process cells for this generation
        if (globalRow > 0 && globalRow < numRows - 1 && globalCol > 0 && globalCol < numCols - 1)
        {
            int numNeighbors = 0;
            numNeighbors += sharedMemory[(sharedRow - 1) * sharedWidth + sharedCol].getStatus();       // Above
            numNeighbors += sharedMemory[(sharedRow + 1) * sharedWidth + sharedCol].getStatus();       // Below
            numNeighbors += sharedMemory[sharedRow * sharedWidth + (sharedCol - 1)].getStatus();       // Left
            numNeighbors += sharedMemory[sharedRow * sharedWidth + (sharedCol + 1)].getStatus();       // Right
            numNeighbors += sharedMemory[(sharedRow - 1) * sharedWidth + (sharedCol - 1)].getStatus(); // Top-left
            numNeighbors += sharedMemory[(sharedRow - 1) * sharedWidth + (sharedCol + 1)].getStatus(); // Top-right
            numNeighbors += sharedMemory[(sharedRow + 1) * sharedWidth + (sharedCol - 1)].getStatus(); // Bottom-left
            numNeighbors += sharedMemory[(sharedRow + 1) * sharedWidth + (sharedCol + 1)].getStatus(); // Bottom-right

            // Apply Game of Life rules
            if (sharedMemory[sharedRow * sharedWidth + sharedCol].getStatus() == 1)
            {
                if (numNeighbors < 2 || numNeighbors > 3)
                    newMatrix[globalRow * numCols + globalCol].setDead();
                else
                    newMatrix[globalRow * numCols + globalCol].setAlive();
            }
            else
            {
                if (numNeighbors == 3)
                    newMatrix[globalRow * numCols + globalCol].setAlive();
                else
                    newMatrix[globalRow * numCols + globalCol].setDead();
            }
        }

        // Synchronize before the next generation
        __syncthreads();

        // Swap matrices only after the entire block completes its work for this generation
        if (gen < maxGens - 1)
        {
            cell *temp = matrix;
            matrix = newMatrix;
            newMatrix = temp;
        }
    }
}

// Function to initialize the matrix on the host remains unchanged
void initializeMatrix(vector<cell> &matrix, int numRows, int numCols)
{
    srand(2);
    for (int i = 0; i < numRows; i++)
    {
        for (int j = 0; j < numCols; j++)
        {
            cell newCell;
            if ((i == 0) || (i == (numRows - 1)) || (j == 0) || (j == (numCols - 1)))
            {
                newCell.setGhost();
            }
            else
            {
                int randomNum = rand() % 2;
                if (randomNum == 0)
                {
                    newCell.setDead();
                }
                else
                {
                    newCell.setAlive();
                }
            }
            matrix[i * numCols + j] = newCell;
        }
    }
}

// Write final matrix to file remains unchanged
void writeToFile(const vector<cell> &matrix, int numRows, int numCols, int maxIterations, int numProcesses, const string &fileDirectory)
{
    ofstream file;
    stringstream fileNameStream;
    fileNameStream << "output." << numRows - 2 << "." << maxIterations << "." << numProcesses << ".cudaimp.txt";
    string fileName = fileNameStream.str();

    file.open(fileDirectory + "/" + fileName);
    if (!file.is_open())
    {
        cerr << "Error: Could not open file " << fileDirectory + "/" + fileName << endl;
        return;
    }

    // Write the matrix excluding the ghost cells (skip the border rows and columns)
    for (int i = 1; i < numRows - 1; i++) // Skip first and last row (ghost cells)
    {
        for (int j = 1; j < numCols - 1; j++) // Skip first and last column (ghost cells)
        {
            file << matrix[i * numCols + j].getStatus() << " ";
        }
        file << endl;
    }
    file.close();

    cout << "File written successfully" << endl;
}

// Main function
int main(int argc, char **argv)
{
    if (argc < 6)
    {
        cerr << "Usage: " << argv[0] << " <numRows> <numCols> <maxGens> <numProcesses> <fileDirectory>" << endl;
        return 1;
    }

    // Parse command-line arguments
    int numRows = atoi(argv[1]) + 2; // Add 2 for ghost cells
    int numCols = atoi(argv[2]) + 2; // Add 2 for ghost cells
    int maxGens = atoi(argv[3]);
    int numProcesses = atoi(argv[4]);
    string fileDirectory = argv[5];

    // Allocate memory for the matrix (flattened 1D array)
    vector<cell> matrix(numRows * numCols);
    vector<cell> newMatrix(numRows * numCols);

    // Initialize matrix with random values and ghost cells
    initializeMatrix(matrix, numRows, numCols);

    // Allocate device memory for matrix and newMatrix
    cell *d_matrix, *d_newMatrix;
    cudaError_t err;
    size_t matrixSize = numRows * numCols * sizeof(cell);

    err = cudaMalloc(&d_matrix, matrixSize);
    if (err != cudaSuccess)
    {
        cerr << "CUDA Error during cudaMalloc for d_matrix: " << cudaGetErrorString(err) << endl;
        return 1;
    }

    err = cudaMalloc(&d_newMatrix, matrixSize);
    if (err != cudaSuccess)
    {
        cerr << "CUDA Error during cudaMalloc for d_newMatrix: " << cudaGetErrorString(err) << endl;
        return 1;
    }

    // Copy data from host to device
    err = cudaMemcpy(d_matrix, matrix.data(), matrixSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cerr << "CUDA Error during cudaMemcpy from host to device: " << cudaGetErrorString(err) << endl;
        return 1;
    }

    // Set up kernel launch dimensions
    dim3 blockSize(16, 16); // Threads per block
    dim3 gridSize((numCols + blockSize.x - 1) / blockSize.x, (numRows + blockSize.y - 1) / blockSize.y);

    // Launch the kernel to process generations
    processGenerations<<<gridSize, blockSize, (blockSize.x + 2) * (blockSize.y + 2) * sizeof(cell)>>>(d_matrix, d_newMatrix, numRows, numCols, maxGens);

    // Check for kernel errors
    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess)
    {
        cerr << "Kernel launch failed: " << cudaGetErrorString(kernelErr) << endl;
        return 1;
    }

    // Ensure the kernel has finished
    cudaDeviceSynchronize();

    // Copy final result back to host
    err = cudaMemcpy(matrix.data(), d_matrix, matrixSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        cerr << "CUDA Error during cudaMemcpy from device to host: " << cudaGetErrorString(err) << endl;
        return 1;
    }

    // Write the final matrix to a file
    writeToFile(matrix, numRows, numCols, maxGens, numProcesses, fileDirectory);

    // Free device memory
    cudaFree(d_matrix);
    cudaFree(d_newMatrix);

    return 0;
}