/*
Wesley Junkins
wcjunkins@crimson.ua.edu
CS 581
Homework 3
To compile: g++ -fopenmp -O -o lifeOMP lifeOMP.cpp OR icpx -qopenmp -O -o lifeOMP lifeOMP.cpp
To run: ./lifeOMP <numRows> <numCols> <numThreads> <maxGens> <fileDirectory> [printGens] [printMatrix]
    Ex: ./lifeOMP 100 100 4 1000 /scratch/$USER
*/

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <fstream>
#include <sstream>

using namespace std;

// A class to represent a cell in the matrix
class cell
{
private:
    bool isGhost;
    int status;

public:
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

// Function to print the matrix
void printMatrix(const vector<vector<cell> >& matrix, int numRows, int numCols)
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
int statusAbove(const vector<vector<cell> >& matrix, int numRows, int numCols, int i, int j)
{
    return matrix[i - 1][j].getStatus();
};

int statusBelow(const vector<vector<cell> >& matrix, int numRows, int numCols, int i, int j)
{
    return matrix[i + 1][j].getStatus();
};

int statusLeft(const vector<vector<cell> >& matrix, int numRows, int numCols, int i, int j)
{
    return matrix[i][j - 1].getStatus();
};

int statusRight(const vector<vector<cell> >& matrix, int numRows, int numCols, int i, int j)
{
    return matrix[i][j + 1].getStatus();
};

int statusAboveLeft(const vector<vector<cell> >& matrix, int numRows, int numCols, int i, int j)
{
    return matrix[i - 1][j - 1].getStatus();
};

int statusAboveRight(const vector<vector<cell> >& matrix, int numRows, int numCols, int i, int j)
{
    return matrix[i - 1][j + 1].getStatus();
};

int statusBelowLeft(const vector<vector<cell> >& matrix, int numRows, int numCols, int i, int j)
{
    return matrix[i + 1][j - 1].getStatus();
};

int statusBelowRight(const vector<vector<cell> >& matrix, int numRows, int numCols, int i, int j)
{
    return matrix[i + 1][j + 1].getStatus();
};

// Function to get the number of neighbors
int getNumNeighbors(const vector<vector<cell> >& matrix, int numRows, int numCols, int i, int j)
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
bool isSameAsLastRound(const vector<vector<cell> >& matrix, const vector<vector<cell> >& newMatrix, int numRows, int numCols)
{
    for(int i = 0; i < numRows; i++)
    {
        for(int j = 0; j < numCols; j++)
        {
            if(matrix[i][j].getStatus() != newMatrix[i][j].getStatus())
            {
                return false;
            };
        };
    };

    return true;
};

// Function to process a section of the matrix
// Returns TRUE if there was a change and FALSE otherwise
bool processSection(int startRow, int endRow, const vector<vector<cell> >& matrix, vector<vector<cell> >& newMatrix, int numRows, int numCols) {
    bool isChange = false;
    for (int i = startRow; i < endRow; i++) {
        for (int j = 0; j < numCols; j++) {
            if (matrix[i][j].getGhostStatus() == true) {
                newMatrix[i][j].setGhost();
            } else {
                int numNeighbors = getNumNeighbors(matrix, numRows, numCols, i, j);
                if (matrix[i][j].getStatus() == 1) {
                    if (numNeighbors == -1) {
                        cout << "Error in logic" << endl;
                    }
                    if (numNeighbors == 0 || numNeighbors == 1 || numNeighbors >= 4) {
                        newMatrix[i][j].setDead();
                        isChange = true;
                    } else {
                        newMatrix[i][j].setAlive();
                    }
                } else {
                    if (numNeighbors == 3) {
                        newMatrix[i][j].setAlive();
                        isChange = true;
                    } else {
                        newMatrix[i][j].setDead();
                    }
                }
            }
        }
    }
    return isChange;
}

// Function to write the final output matrix to a file
void writeToFile(const vector<vector<cell>>& matrix, int numRows, int numCols, int maxIterations, int numThreads, const string& fileDirectory) {
    ofstream file;
    stringstream fileNameStream;
    fileNameStream << "output." << numRows << "." << maxIterations << "." << numThreads << ".txt";
    string fileName = fileNameStream.str();
    
    file.open(fileDirectory + "/" + fileName);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << fileDirectory + "/" + fileName << endl;
        return;
    }

    for (int i = 1; i < numRows - 1; i++) {
        for (int j = 1; j < numCols - 1; j++) {
            file << matrix[i][j].getStatus() << " ";
        }
        file << endl;
    }
    file.close();
}

// Main function
int main(int argc, char** argv) {
    if (argc < 6) {
        cerr << "Usage: " << argv[0] << " <numRows> <numCols> <numThreads> <maxGens> <fileDirectory> [printGens] [printMatrix]" << endl;
        return 1;
    }
    int numRows = atoi(argv[1]);
    int numCols = atoi(argv[2]);
    int numThreads = atoi(argv[3]);
    int maxGens = atoi(argv[4]);
    string fileDirectory = argv[5];

    vector<vector<cell>> matrix(numRows, vector<cell>(numCols));
    vector<vector<cell>> newMatrix(numRows, vector<cell>(numCols));
    //srand(time(0));
    srand(2);

    // Populate the initial matrix
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            cell newCell;
            if ((i == 0) || (i == (numRows - 1)) || (j == 0) || (j == (numCols - 1))) {
                newCell.setGhost();
            } else {
                int randomNum = rand() % 2;
                if (randomNum == 0) {
                    newCell.setDead();
                } else {
                    newCell.setAlive();
                }
            }
            matrix[i][j] = newCell;
        }
    }

    int threadID;
    int rowsForEachThread;
    int startRow;
    int endRow;
    int runTimes;
    bool exitFlag = false;
    bool isChangeGlobal = false;

    // OpenMP Parallel region (created once for the entire simulation)
    #pragma omp parallel default(none) shared(matrix, newMatrix, numRows, numCols, numThreads, maxGens, argc, argv, cout, fileDirectory, exitFlag, isChangeGlobal) private(threadID, rowsForEachThread, startRow, endRow, runTimes) num_threads(numThreads)
    {
        threadID = omp_get_thread_num();
        rowsForEachThread = numRows / numThreads;
        startRow = threadID * rowsForEachThread;
        endRow = (threadID == numThreads - 1) ? numRows : startRow + rowsForEachThread;

        for (int runTimes = 0; runTimes < maxGens; runTimes++) {
            bool isChangeLocal = false; // Flag to check if there was a change in the matrix for this thread only

            // Each thread processes its assigned rows
            for (int i = startRow; i < endRow; i++) {
                isChangeLocal |= processSection(i, i + 1, matrix, newMatrix, numRows, numCols);
            }

            // Enter a critical section to update the global flag
            #pragma omp critical
            {
                isChangeGlobal |= isChangeLocal;
            }

            // Synchronize threads at the end of each generation
            #pragma omp barrier

            // Check if there was a change in the matrix global matrix
            #pragma omp single
            {
                if(!isChangeGlobal) {
                    exitFlag = true; // No global changes, so set the exit flag to true so we can exit
                } else
                {
                    isChangeGlobal = false; // Reset the global flag for the next generation because we found at least one change globally
                }

                #pragma omp flush(exitFlag)
            }

            // Exit if there was no change in the matrix
            #pragma omp barrier
            if (exitFlag) {
                #pragma omp single
                {
                    cout<<"Process exiting at generation: "<<runTimes<<endl;
                }
                break;
            }

            // Print information if required
            if (argc > 6 && string(argv[6]) == "printGens") {
                #pragma omp single
                cout << runTimes << endl;
            }

            if (argc > 6 && string(argv[6]) == "printMatrix") {
                #pragma omp single
                printMatrix(matrix, numRows, numCols);
            }

            // Swap matrices
            #pragma omp single
            matrix = newMatrix;

        }
    }
    writeToFile(matrix, numRows, numCols, maxGens, numThreads, fileDirectory);
    return 0;
}