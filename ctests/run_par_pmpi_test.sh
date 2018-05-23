#!/bin/sh
# This is a test script for the GPTL package. It tests the parallel
# functionality with the MPI library.
#
# Ed Hartnett 5/23/18

set -e
echo
echo "Testing MPI functionality..."
mpiexec -n 2 ./pmpi
echo "SUCCESS!"
exit 0
