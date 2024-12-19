/*
Wesley Junkins
wcjunkins@crimson.ua.edu
CS 581
Homework 4
To compile: mpic++ lifeMPInoblock.cpp -O -o lifeMPInoblock -std=c++11
To run: mpirun -n <numProcesses> ./lifeMPInoblock <numRows> <numCols> <maxGens> <numProcesses> <fileDirectory>
*/

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <mpi.h>
#include <fstream>
#include <sstream>

using namespace std;

// A class to represent a cell in the matrix
class cell
{
public:
    bool isGhost;
    int status;

    cell()
    {
        isGhost = false;
        status = 0;
    }

    void setAlive()
    {
        if (this->isGhost == false)
        {
            this->status = 1;
        };
    };

    void setDead()
    {
        this->status = 0;
    };

    int getStatus() const
    {
        return this->status;
    };

    void setGhost()
    {
        this->isGhost = true;
        this->status = 0;
    };

    bool getGhostStatus() const
    {
        return this->isGhost;
    };
};

// void printer(int rank, string message)
// {
//     cout << "Rank " << rank << ": " << message << endl;
// };

// Function to print the matrix
void printMatrix(const vector<vector<cell>> &matrix, int numRows, int numCols)
{
#ifdef _WIN32
    system("cls");
#else
    system("clear");
#endif

    for (int i = 0; i < numRows; i++)
    {
        for (int j = 0; j < numCols; j++)
        {
            cout << matrix.at(i).at(j).getStatus() << " ";
        };
        cout << endl;
    };
};

// Functions to get the status of the neighbors
int statusAbove(const vector<vector<cell>> &matrix, int numRows, int numCols, int i, int j)
{
    return matrix[i - 1][j].getStatus();
};

int statusBelow(const vector<vector<cell>> &matrix, int numRows, int numCols, int i, int j)
{
    return matrix[i + 1][j].getStatus();
};

int statusLeft(const vector<vector<cell>> &matrix, int numRows, int numCols, int i, int j)
{
    return matrix[i][j - 1].getStatus();
};

int statusRight(const vector<vector<cell>> &matrix, int numRows, int numCols, int i, int j)
{
    return matrix[i][j + 1].getStatus();
};

int statusAboveLeft(const vector<vector<cell>> &matrix, int numRows, int numCols, int i, int j)
{
    return matrix[i - 1][j - 1].getStatus();
};

int statusAboveRight(const vector<vector<cell>> &matrix, int numRows, int numCols, int i, int j)
{
    return matrix[i - 1][j + 1].getStatus();
};

int statusBelowLeft(const vector<vector<cell>> &matrix, int numRows, int numCols, int i, int j)
{
    return matrix[i + 1][j - 1].getStatus();
};

int statusBelowRight(const vector<vector<cell>> &matrix, int numRows, int numCols, int i, int j)
{
    return matrix[i + 1][j + 1].getStatus();
};

// Function to get the number of neighbors
int getNumNeighbors(const vector<vector<cell>> &matrix, int numRows, int numCols, int i, int j)
{
    if (matrix[i][j].getGhostStatus() == false)
    {
        return (statusAbove(matrix, numRows, numCols, i, j) + statusBelow(matrix, numRows, numCols, i, j) + statusLeft(matrix, numRows, numCols, i, j) + statusRight(matrix, numRows, numCols, i, j) + statusAboveLeft(matrix, numRows, numCols, i, j) + statusAboveRight(matrix, numRows, numCols, i, j) + statusBelowLeft(matrix, numRows, numCols, i, j) + statusBelowRight(matrix, numRows, numCols, i, j));
    }
    else
    {
        return -1;
    };
};

// Function to check if the matrix is the same as the last round (Did not use this function in the final code)
bool isSameAsLastRound(const vector<vector<cell>> &matrix, const vector<vector<cell>> &newMatrix, int numRows, int numCols)
{
    for (int i = 0; i < numRows; i++)
    {
        for (int j = 0; j < numCols; j++)
        {
            if (matrix[i][j].getStatus() != newMatrix[i][j].getStatus())
            {
                return false;
            };
        };
    };

    return true;
};

// Function to process a section of the matrix
// Each process works only on its assigned rows (from startRow to endRow)
bool processSection(int startRow, int endRow, const vector<vector<cell>> &matrix, vector<vector<cell>> &newMatrix, int numRows, int numCols, int rank)
{
    bool isChange = false;

    for (int i = startRow; i < endRow; i++)
    {
        for (int j = 1; j < numCols - 1; j++) // Skip ghost columns at the boundaries
        {
            bool isGhost = matrix[i][j].getGhostStatus();
            if (isGhost)
            {
                newMatrix[i][j].setGhost();
                continue; // Skip further processing for ghost cells
            }

            // Calculate the number of live neighbors
            int numNeighbors =
                matrix[i - 1][j - 1].getStatus() + matrix[i - 1][j].getStatus() + matrix[i - 1][j + 1].getStatus() +
                matrix[i][j - 1].getStatus() + matrix[i][j + 1].getStatus() +
                matrix[i + 1][j - 1].getStatus() + matrix[i + 1][j].getStatus() + matrix[i + 1][j + 1].getStatus();

            // Process the cell based on the number of live neighbors
            if (matrix[i][j].getStatus() == 1)
            {
                if (numNeighbors < 2 || numNeighbors > 3)
                {
                    newMatrix[i][j].setDead();
                    isChange = true;
                }
                else
                {
                    newMatrix[i][j].setAlive(); // Cell stays alive
                }
            }
            else
            {
                if (numNeighbors == 3)
                {
                    newMatrix[i][j].setAlive();
                    isChange = true;
                }
                else
                {
                    newMatrix[i][j].setDead();
                }
            }
        }
    }

    return isChange;
}

// Function to write the final output matrix to a file
void writeToFile(const vector<vector<cell>> &matrix, int numRows, int numCols, int maxIterations, int numProcesses, const string &fileDirectory)
{
    ofstream file;
    stringstream fileNameStream;
    fileNameStream << "output." << numRows - 2 << "." << maxIterations << "." << numProcesses << ".txt";
    string fileName = fileNameStream.str();

    file.open(fileDirectory + "/" + fileName);
    if (!file.is_open())
    {
        cerr << "Error: Could not open file " << fileDirectory + "/" + fileName << endl;
        return;
    }

    for (int i = 1; i < numRows - 1; i++)
    {
        for (int j = 1; j < numCols - 1; j++)
        {
            file << matrix[i][j].getStatus() << " ";
        }
        file << endl;
    }
    file.close();
}

// Main function
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    if (argc < 6)
    {
        cerr << "Usage: " << argv[0] << " <numRows> <numCols> <maxGens> <numProcesses> <fileDirectory>" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int numRows = atoi(argv[1]) + 2;
    int numCols = atoi(argv[2]) + 2;
    int maxGens = atoi(argv[3]);
    int numProcesses = atoi(argv[4]);
    string fileDirectory = argv[5];

    // Create a custome MPI datatype since my cells are objects, not primitives.
    MPI_Datatype MPI_CELL;
    int blockLengths[2] = {1, 1};
    MPI_Datatype types[2] = {MPI_CXX_BOOL, MPI_INT};
    MPI_Aint offsets[2] = {offsetof(cell, isGhost), offsetof(cell, status)};
    MPI_Type_create_struct(2, blockLengths, offsets, types, &MPI_CELL);
    MPI_Type_commit(&MPI_CELL);

    // Get the rank and size of the MPI world
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // printer(rank, "Entered Program");

    // Divide matrix rows among the processes
    int rowsPerProcess = numRows / size;
    int extraRows = numRows % size;

    // Calculate start and end rows for each process
    int startRow = rank * rowsPerProcess + min(rank, extraRows);
    int endRow = startRow + rowsPerProcess + (rank < extraRows ? 1 : 0);

    // Adjust for ghost cells
    // May not use these
    int effectiveStartRow = startRow;
    int effectiveEndRow = endRow;
    if (rank > 0)
        effectiveStartRow--; // Include ghost row above
    if (rank < size - 1)
        effectiveEndRow++; // Include ghost row below

    vector<vector<cell>> matrix(numRows, vector<cell>(numCols));
    vector<vector<cell>> newMatrix(numRows, vector<cell>(numCols));
    // srand(time(0));
    srand(2);

    // printer(rank, "Matrix Created, about to populate initial matrix");

    // Populate the initial matrix
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
            matrix[i][j] = newCell;
        }
    }

    // printMatrix(matrix, numRows, numCols);

    // printer(rank, "Initial matrix populated, about to start the main iteration loop");

    // Main iteration loop
    bool globalChange = false;
    for (int gen = 0; gen < maxGens; gen++)
    {
        // printer(rank, "Entered Generation Loop");

        // Each process takes care of the rows that were assigned to it
        bool localChange = processSection(startRow, endRow, matrix, newMatrix, numRows, numCols, rank);

        // printer(rank, "Processed Section, about to exchange ghost cells");

        // Exchange ghost cell borders with neighbors using MPI Isend and Irecv functions
        // If there is no process above or below it, then do not send anything
        MPI_Request requests[4];
        int req_count = 0;

        if (rank > 0)
        {
            // Send to upper neighbor and receive from them
            MPI_Isend(&newMatrix[startRow][0], numCols, MPI_CELL, rank - 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
            MPI_Irecv(&newMatrix[startRow - 1][0], numCols, MPI_CELL, rank - 1, 1, MPI_COMM_WORLD, &requests[req_count++]);
        }
        if (rank < size - 1)
        {
            // Send to lower neighbor and receive from them
            MPI_Isend(&newMatrix[endRow - 1][0], numCols, MPI_CELL, rank + 1, 1, MPI_COMM_WORLD, &requests[req_count++]);
            MPI_Irecv(&newMatrix[endRow][0], numCols, MPI_CELL, rank + 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
        }
        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);

        // printer(rank, "Exchanged Ghost Cells, about to check for global change");

        // Get the global change flag from all processes
        MPI_Allreduce(&localChange, &globalChange, 1, MPI_CXX_BOOL, MPI_LOR, MPI_COMM_WORLD);

        // printer(rank, "Checked for global change, about to swap matrices");

        // Swap matrices for next generation
        std::swap(matrix, newMatrix);

        // printer(rank, "Swapped matrices, about to print matrix if there was no global change");

        // Terminate the program if no global changes were made
        if (!globalChange)
        {
            // printer(rank, "No global change, breaking out of loop");
            if (rank == 0)
            {
                cout << "Program exiting at generation: " << gen << endl;
            }
            break;
        }
    }

    // printer(rank, "Exited Generation Loop, about to gather results");

    // Gather results on the root process
    // Used Gatherv since there are a variable number of rows per process
    int localRowCount = endRow - startRow;
    vector<cell> localData(localRowCount * numCols);
    for (int i = 0; i < localRowCount; i++)
    {
        for (int j = 0; j < numCols; j++)
        {
            localData[i * numCols + j] = matrix[startRow + i][j];
        }
    }
    vector<int> counts(size);
    vector<int> displacements(size);
    if (rank == 0)
    {
        counts.resize(size);
        displacements.resize(size);
        int offset = 0;
        for (int i = 0; i < size; ++i)
        {
            int rowsForProcess = (i < extraRows) ? rowsPerProcess + 1 : rowsPerProcess;
            counts[i] = rowsForProcess * numCols;
            displacements[i] = offset;
            offset += counts[i];
        }
    }
    vector<cell> fullMatrix(numRows * numCols);
    MPI_Gatherv(localData.data(), localData.size(), MPI_CELL, fullMatrix.data(), counts.data(), displacements.data(), MPI_CELL, 0, MPI_COMM_WORLD);

    // printer(rank, "Gathered results, about to write to file");

    // Root process writes the gathered data to file
    if (rank == 0)
    {
        // printer(rank, "Writing to file");

        vector<vector<cell>> gatheredMatrix(numRows, vector<cell>(numCols));
        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numCols; j++)
            {
                gatheredMatrix[i][j] = fullMatrix[i * numCols + j];
            }
        }
        writeToFile(gatheredMatrix, numRows, numCols, maxGens, numProcesses, fileDirectory);

        // printer(rank, "Finished writing to file");
    }

    MPI_Type_free(&MPI_CELL);

    MPI_Finalize();

    // printer(rank, "Exited Program");

    return 0;
}