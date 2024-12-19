Homework 4
Wesley Junkins
CS581

This repo has 3 folders:

- Source code: contains both C++ source code files that were compiled and run to produce the output.

- Shell Scripts: contains 2 shell scripts:
    - lifeMPIscript.sh: this script runs 3 instances of the blocking and nonblocking code at each processor level and displays the time taken to run each.
    - checkScript.sh: the script is used to compare the outputs among all the MPI versions to ensure accuracy.

- SLURM IO Files: this contains the SLURM input and output files for each of the scripts I ran. ("res" for the results of the lifeMPIscript and "resCHECK" for the results of the checkScript).