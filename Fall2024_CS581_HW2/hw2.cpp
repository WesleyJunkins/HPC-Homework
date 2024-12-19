/*
Wesley Junkins
wcjunkins@crimson.ua.edu
CS 581
Homework #2 (Code from HW#1)
To compile: g++ life.cpp -O -std=c++11
To run: ./a.out 1000 1000 5000
  or    ./a.out 1000 1000 5000 printMatrix
  or    ./a.out 1000 1000 5000 printGens
*/

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

// Cell class
// This class represents a cell in the Game of Life matrix. It has two attributes: isGhost and status
// isGhost determines if the cell is a ghost cell
// status determines if the cell is alive or dead (1 or 0, respectively)
// getter and setter functions allow for easy interfacing with a cell object
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

// Print the matrix (including ghost cells) to the terminal in a grid-pattern
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

// Print the generation number (or round number)
void printRound(int roundNum)
{
    #ifdef _WIN32
        system("cls");
    #else
        system("clear");
    #endif

    cout<<roundNum;
};

// Check the status of the cell above the current cell
int statusAbove(const vector<vector<cell> >& matrix, int numRows, int numCols, int i, int j)
{
    return matrix[i - 1][j].getStatus();
};

// Check the status of the cell below the current cell
int statusBelow(const vector<vector<cell> >& matrix, int numRows, int numCols, int i, int j)
{
    return matrix[i + 1][j].getStatus();
};

// Check the status of the cell to the left of the current cell
int statusLeft(const vector<vector<cell> >& matrix, int numRows, int numCols, int i, int j)
{
    return matrix[i][j - 1].getStatus();
};

// Check the status of the cell to the right of the current cell
int statusRight(const vector<vector<cell> >& matrix, int numRows, int numCols, int i, int j)
{
    return matrix[i][j + 1].getStatus();
};

// Check the status of the cell above and to the left of the current cell
int statusAboveLeft(const vector<vector<cell> >& matrix, int numRows, int numCols, int i, int j)
{
    return matrix[i - 1][j - 1].getStatus();
};

// Check the status of the cell above and to the right of the current cell
int statusAboveRight(const vector<vector<cell> >& matrix, int numRows, int numCols, int i, int j)
{
    return matrix[i - 1][j + 1].getStatus();
};

// Check the status of the cell below and to the left of the current cell
int statusBelowLeft(const vector<vector<cell> >& matrix, int numRows, int numCols, int i, int j)
{
    return matrix[i + 1][j - 1].getStatus();
};

// Check the status of the cell below and to the right of the current cell
int statusBelowRight(const vector<vector<cell> >& matrix, int numRows, int numCols, int i, int j)
{
    return matrix[i + 1][j + 1].getStatus();
};

// Check the status of all the neighboring cells of the current cell using the above functions
// Returns the integer number of alive neighbors
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

// Compare the last two matrices to see if they are the same
// Returns true if there was no change
// This function was scrapped in a revision
// bool isSameAsLastRound(const vector<vector<cell> >& matrix, const vector<vector<cell> >& newMatrix, int numRows, int numCols)
// {
//     for(int i = 0; i < numRows; i++)
//     {
//         for(int j = 0; j < numCols; j++)
//         {
//             if(matrix[i][j].getStatus() != newMatrix[i][j].getStatus())
//             {
//                 return false;
//             };
//         };
//     };

//     return true;
// };

int main(int argc, char *argv[])
{
    // Command line arguments
    int numRows = atoi(argv[1]) + 2;
    int numCols = atoi(argv[2]) + 2;
    int maxGens = atoi(argv[3]);

    // Create the two matrices
    vector<vector<cell> > matrix(numRows, vector<cell>(numCols));
    vector<vector<cell> > newMatrix(numRows, vector<cell>(numCols));

    // Seed the random number generator to use later
    srand(time(0));

    // Populate the initial matrix
    for (int i = 0; i < numRows; i++)
    {
        for (int j = 0; j < numCols; j++)
        {
            cell newCell;
            if ((i == 0) || (i == (numRows - 1)) || (j == 0) || (j == (numCols - 1)))
            {
                // This is a ghost cell
                newCell.setGhost();
            }
            else
            {
                // This is NOT a ghost cell
                int randomNum = rand() % 2;
                if (randomNum == 0)
                {
                    // Cell is dead (0)
                    newCell.setDead();
                }
                else
                {
                    // Cell is alive (1)
                    newCell.setAlive();
                };
            };
            matrix[i][j] = newCell;
        };
    };

    // Run through the specified number of generations
    for (int runTimes = 0; runTimes < maxGens; runTimes++)
    {
        bool isChange = false;
        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numCols; j++)
            {
                if (matrix[i][j].getGhostStatus() == true)
                {
                    // This is a ghost cell
                    newMatrix[i][j].setGhost();
                }
                else
                {
                    // This is NOT a ghost cell
                    // Check status
                    if (matrix[i][j].getStatus() == 1)
                    {
                        // This cell is alive
                        int numNeighbors = getNumNeighbors(matrix, numRows, numCols, i, j);

                        if (numNeighbors == -1)
                        {
                            cout << "Error in logic" << endl;
                            return 0;
                        };

                        if ((numNeighbors == 0) || (numNeighbors == 1) || (numNeighbors >= 4))
                        {
                            // Cell has 1 or no neighbors, or it has 4 or more neighbors, so it dies in the next generation
                            newMatrix[i][j].setDead();
                            isChange = true;
                        }
                        else
                        {
                            // No change
                            newMatrix[i][j].setAlive();
                        }
                    }
                    else
                    {
                        // This cell is dead
                        int numNeighbors = getNumNeighbors(matrix, numRows, numCols, i, j);
                        if (numNeighbors == 3)
                        {
                            // Cell has exactly 3 neighbors, so it resurrects in the next generation
                            newMatrix[i][j].setAlive();
                            isChange = true;
                        }
                        else
                        {
                            // No change
                            newMatrix[i][j].setDead();
                        };
                    };
                };
            };
        };

        // Print the current generation number (if instructed)
        if( argc > 4 && string(argv[4]) == "printGens")
        {
            cout<<runTimes<<endl;
        };
        
        // Print the current matrix (if specified)
        if (argc > 4 && string(argv[4]) == "printMatrix") {
            printMatrix(matrix, numRows, numCols);
        };

        // This code was revised
        // if(isSameAsLastRound(matrix, newMatrix, numRows, numCols))
        // {
        //     return 0;
        // };

        // If there was no change within this generation, end the program early
        if(!isChange)
        {
            return 0;
        };

        // The new matrix becomes the current matrix for the next round
        matrix = newMatrix;
    };

    return 0;
};