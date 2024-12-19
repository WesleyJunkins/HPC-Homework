// By Wesley Junkins
// wcjunkins@crimson.ua.edu
//
// Homework 5: CUDA Game of Life (CS 581)
// This program implements the Game of Life using CUDA.
//
// How to compile: nvcc lifeCUDA.cu -o lifeCUDA
// How to run: ./lifeCUDA <numRows> <numCols> <maxGens> <numProcesses> <fileDirectory>

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

// Kernel function to process the entire matrix for a specified number of generations
// Kernel function to process the entire matrix for a specified number of generations
__global__ void processGenerations(cell *matrix, cell *newMatrix, int numRows, int numCols, int maxGens)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    // Bounds check for global index
    if (index >= numRows * numCols)
        return;

    int i = index / numCols; // Row index
    int j = index % numCols; // Column index

    // Skip ghost cells
    if (i <= 0 || i >= numRows - 1 || j <= 0 || j >= numCols - 1)
    {
        newMatrix[index].setGhost(); // Ensure ghost cells remain unchanged
        return;
    }

    for (int gen = 0; gen < maxGens; gen++)
    {
        int numNeighbors = 0;

        // Count neighbors
        numNeighbors += matrix[(i - 1) * numCols + j].getStatus();       // Above
        numNeighbors += matrix[(i + 1) * numCols + j].getStatus();       // Below
        numNeighbors += matrix[i * numCols + (j - 1)].getStatus();       // Left
        numNeighbors += matrix[i * numCols + (j + 1)].getStatus();       // Right
        numNeighbors += matrix[(i - 1) * numCols + (j - 1)].getStatus(); // Top-left
        numNeighbors += matrix[(i - 1) * numCols + (j + 1)].getStatus(); // Top-right
        numNeighbors += matrix[(i + 1) * numCols + (j - 1)].getStatus(); // Bottom-left
        numNeighbors += matrix[(i + 1) * numCols + (j + 1)].getStatus(); // Bottom-right

        // Apply Game of Life rules
        if (matrix[index].getStatus() == 1)
        {
            if (numNeighbors < 2 || numNeighbors > 3)
            {
                newMatrix[index].setDead();
            }
            else
            {
                newMatrix[index].setAlive();
            }
        }
        else
        {
            if (numNeighbors == 3)
            {
                newMatrix[index].setAlive();
            }
            else
            {
                newMatrix[index].setDead();
            }
        }

        // Swap matrices for the next generation
        cell *temp = matrix;
        matrix = newMatrix;
        newMatrix = temp;

        // Synchronization is not necessary because we alternate buffers
    }
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

// Function to write the final output matrix to a file
void writeToFile(const vector<cell> &matrix, int numRows, int numCols, int maxIterations, int numProcesses, const string &fileDirectory)
{
    ofstream file;
    stringstream fileNameStream;
    fileNameStream << "output." << numRows - 2 << "." << maxIterations << "." << numProcesses << ".cuda.txt";
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

    // Initialize the matrix
    initializeMatrix(matrix, numRows, numCols);

    // Allocate device memory
    cell *d_matrix;
    cell *d_newMatrix;
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

    // Calculate the total number of elements in the matrix
    int totalElements = numRows * numCols;

    // Use 1024 threads per block
    int threadsPerBlock = 32 * 32;

    // Calculate the number of blocks
    int numBlocks = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

    // Execute kernel to process all generations in one go
    processGenerations<<<numBlocks, threadsPerBlock>>>(d_matrix, d_newMatrix, numRows, numCols, maxGens);
    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess)
    {
        cerr << "Kernel launch failed: " << cudaGetErrorString(kernelErr) << endl;
        return 1;
    }

    // Ensure the kernel finishes before moving on
    cudaDeviceSynchronize();

    // Copy the results back to host
    err = cudaMemcpy(&matrix[0], d_matrix, matrixSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        cerr << "CUDA Error during cudaMemcpy from device to host: " << cudaGetErrorString(err) << endl;
        return 1;
    }

    // Write final output to file
    writeToFile(matrix, numRows, numCols, maxGens, numProcesses, fileDirectory);

    // Clean up device memory
    cudaFree(d_matrix);
    cudaFree(d_newMatrix);

    return 0;
}
