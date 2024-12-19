// By Wesley Junkins
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

// A class to represent a cell in the matrix
class cell
{
public:
    bool isGhost;
    int status;

    __host__ __device__ cell() // Constructor should be available both on the host and device
    {
        isGhost = false;
        status = 0;
    }

    __host__ __device__ void setAlive()
    {
        if (!this->isGhost)
        {
            this->status = 1;
        }
    }

    __host__ __device__ void setDead()
    {
        this->status = 0;
    }

    __host__ __device__ int getStatus() const
    {
        return this->status;
    }

    __host__ __device__ void setGhost()
    {
        this->isGhost = true;
        this->status = 0;
    }

    __host__ __device__ bool getGhostStatus() const
    {
        return this->isGhost;
    }
};

// Kernel function to process the matrix using shared memory
__global__ void processSection(cell *matrix, cell *newMatrix, int numRows, int numCols)
{
    extern __shared__ cell sharedMemory[]; // Shared memory for the block

    int globalRow = blockIdx.y * blockDim.y + threadIdx.y;
    int globalCol = blockIdx.x * blockDim.x + threadIdx.x;

    int sharedRow = threadIdx.y + 1;
    int sharedCol = threadIdx.x + 1;

    // Load the current cell into shared memory
    if (globalRow < numRows && globalCol < numCols)
    {
        sharedMemory[sharedRow * (blockDim.x + 2) + sharedCol] = matrix[globalRow * numCols + globalCol];
    }

    // Load ghost cells (ensure proper boundary handling)
    if (threadIdx.y == 0 && globalRow > 0) // Top ghost row
    {
        sharedMemory[(sharedRow - 1) * (blockDim.x + 2) + sharedCol] = matrix[(globalRow - 1) * numCols + globalCol];
    }
    if (threadIdx.y == blockDim.y - 1 && globalRow < numRows - 1) // Bottom ghost row
    {
        sharedMemory[(sharedRow + 1) * (blockDim.x + 2) + sharedCol] = matrix[(globalRow + 1) * numCols + globalCol];
    }
    if (threadIdx.x == 0 && globalCol > 0) // Left ghost column
    {
        sharedMemory[sharedRow * (blockDim.x + 2) + (sharedCol - 1)] = matrix[globalRow * numCols + (globalCol - 1)];
    }
    if (threadIdx.x == blockDim.x - 1 && globalCol < numCols - 1) // Right ghost column
    {
        sharedMemory[sharedRow * (blockDim.x + 2) + (sharedCol + 1)] = matrix[globalRow * numCols + (globalCol + 1)];
    }

    // Diagonal ghost cells
    if (threadIdx.y == 0 && threadIdx.x == 0 && globalRow > 0 && globalCol > 0) // Top-left diagonal
    {
        sharedMemory[(sharedRow - 1) * (blockDim.x + 2) + (sharedCol - 1)] = matrix[(globalRow - 1) * numCols + (globalCol - 1)];
    }
    if (threadIdx.y == 0 && threadIdx.x == blockDim.x - 1 && globalRow > 0 && globalCol < numCols - 1) // Top-right diagonal
    {
        sharedMemory[(sharedRow - 1) * (blockDim.x + 2) + (sharedCol + 1)] = matrix[(globalRow - 1) * numCols + (globalCol + 1)];
    }
    if (threadIdx.y == blockDim.y - 1 && threadIdx.x == 0 && globalRow < numRows - 1 && globalCol > 0) // Bottom-left diagonal
    {
        sharedMemory[(sharedRow + 1) * (blockDim.x + 2) + (sharedCol - 1)] = matrix[(globalRow + 1) * numCols + (globalCol - 1)];
    }
    if (threadIdx.y == blockDim.y - 1 && threadIdx.x == blockDim.x - 1 && globalRow < numRows - 1 && globalCol < numCols - 1) // Bottom-right diagonal
    {
        sharedMemory[(sharedRow + 1) * (blockDim.x + 2) + (sharedCol + 1)] = matrix[(globalRow + 1) * numCols + (globalCol + 1)];
    }

    __syncthreads(); // Ensure all threads have loaded shared memory

    // Process the current cell
    if (globalRow > 0 && globalRow < numRows - 1 && globalCol > 0 && globalCol < numCols - 1)
    {
        int numNeighbors = 0;

        // Count neighbors
        numNeighbors += sharedMemory[(sharedRow - 1) * (blockDim.x + 2) + sharedCol].getStatus();       // Above
        numNeighbors += sharedMemory[(sharedRow + 1) * (blockDim.x + 2) + sharedCol].getStatus();       // Below
        numNeighbors += sharedMemory[sharedRow * (blockDim.x + 2) + (sharedCol - 1)].getStatus();       // Left
        numNeighbors += sharedMemory[sharedRow * (blockDim.x + 2) + (sharedCol + 1)].getStatus();       // Right
        numNeighbors += sharedMemory[(sharedRow - 1) * (blockDim.x + 2) + (sharedCol - 1)].getStatus(); // Top-left diagonal
        numNeighbors += sharedMemory[(sharedRow - 1) * (blockDim.x + 2) + (sharedCol + 1)].getStatus(); // Top-right diagonal
        numNeighbors += sharedMemory[(sharedRow + 1) * (blockDim.x + 2) + (sharedCol - 1)].getStatus(); // Bottom-left diagonal
        numNeighbors += sharedMemory[(sharedRow + 1) * (blockDim.x + 2) + (sharedCol + 1)].getStatus(); // Bottom-right diagonal

        // Apply Game of Life rules
        if (sharedMemory[sharedRow * (blockDim.x + 2) + sharedCol].getStatus() == 1)
        {
            if (numNeighbors < 2 || numNeighbors > 3)
            {
                newMatrix[globalRow * numCols + globalCol].setDead();
            }
            else
            {
                newMatrix[globalRow * numCols + globalCol].setAlive();
            }
        }
        else
        {
            if (numNeighbors == 3)
            {
                newMatrix[globalRow * numCols + globalCol].setAlive();
            }
            else
            {
                newMatrix[globalRow * numCols + globalCol].setDead();
            }
        }
    }

    __syncthreads(); // Final sync before exiting
}

// Function to initialize the matrix on the host
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

// // Function to print the matrix (2D format)
// void printMatrix(const vector<cell> &matrix, int numRows, int numCols)
// {
//     cout << "Matrix:" << endl;
//     for (int i = 0; i < numRows; i++)
//     {
//         for (int j = 0; j < numCols; j++)
//         {
//             cout << matrix[i * numCols + j].getStatus() << " ";
//         }
//         cout << endl;
//     }
// }

// Function to write the final output matrix to a file
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

    int numRows = atoi(argv[1]) + 2; // Add 2 for ghost cells
    int numCols = atoi(argv[2]) + 2; // Add 2 for ghost cells
    int maxGens = atoi(argv[3]);
    int numProcesses = atoi(argv[4]);
    string fileDirectory = argv[5];

    // Allocate memory for the matrix as a flattened 1D array
    vector<cell> matrix(numRows * numCols);
    vector<cell> newMatrix(numRows * numCols);

    // Initialize the matrix
    initializeMatrix(matrix, numRows, numCols);

    // Print the matrix
    // printMatrix(matrix, numRows, numCols);

    // Allocate device memory
    cell *d_matrix, *d_newMatrix;
    size_t matrixSize = numRows * numCols * sizeof(cell);

    cudaError_t err;
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
    err = cudaMemcpy(d_matrix, &matrix[0], matrixSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cerr << "CUDA Error during cudaMemcpy from host to device: " << cudaGetErrorString(err) << endl;
        return 1;
    }

    // Use 16x16 block size
    int threadsPerBlock = 16;
    dim3 threads(threadsPerBlock, threadsPerBlock); // 2D grid for thread blocks

    // Define the grid size based on matrix dimensions and block size
    dim3 grid((numCols + threadsPerBlock - 1) / threadsPerBlock, (numRows + threadsPerBlock - 1) / threadsPerBlock);

    // Allocate enough shared memory for each block
    int sharedMemSize = sizeof(cell) * (threadsPerBlock + 2) * (threadsPerBlock + 2);

    // Main iteration loop
    for (int gen = 0; gen < maxGens; gen++)
    {
        // cout << "Running Generation: " << gen << endl;

        // Launch kernel with multiple blocks and threads
        processSection<<<grid, threads, sharedMemSize>>>(d_matrix, d_newMatrix, numRows, numCols);
        cudaError_t kernelErr = cudaGetLastError();
        if (kernelErr != cudaSuccess)
        {
            cerr << "Kernel launch failed: " << cudaGetErrorString(kernelErr) << endl;
            return 1;
        }

        // Ensure the kernel finishes before moving on
        cudaDeviceSynchronize();

        // Copy the results back to host
        err = cudaMemcpy(&matrix[0], d_newMatrix, matrixSize, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            cerr << "CUDA Error during cudaMemcpy from device to host: " << cudaGetErrorString(err) << endl; // THE ERROR HAPPENS HERE!!!
            return 1;
        }

        // Print the matrix
        // printMatrix(matrix, numRows, numCols);

        // Swap matrices
        cell *temp = d_matrix;
        d_matrix = d_newMatrix;
        d_newMatrix = temp;
        cudaDeviceSynchronize(); // Ensure all operations are complete before swapping
    }

    // Write the final matrix to a file
    writeToFile(matrix, numRows, numCols, maxGens, numProcesses, fileDirectory);

    // Free device memory
    if (d_matrix != nullptr)
    {
        cudaFree(d_matrix);
    }
    if (d_newMatrix != nullptr)
    {
        cudaFree(d_newMatrix);
    }

    // cout << "END" << endl;

    return 0;
}