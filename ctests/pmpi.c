#include <stdio.h>
#include <mpi.h>
#include "../gptl.h"

int main (int argc, char **argv)
{
  int i, ret, iam;
  int size;
  const int count = 1024;
  const MPI_Comm comm = MPI_COMM_WORLD;
  const int tag = 98;
  double sendbuf[count];
  double recvbuf[count];
  MPI_Status status;
  MPI_Request request;
  int dest;
  int source;
  int DebugWait = 1;

  //  while (DebugWait) {
  //  }

  ret = MPI_Init (&argc, &argv);               // Initialize MPI
  ret = MPI_Comm_rank (comm, &iam);  // Get my rank
  ret = MPI_Comm_size (comm, &size); // Get communicator size

  ret = GPTLsetoption (GPTLoverhead, 0);       // Don't print overhead stats
  ret = GPTLsetoption (GPTLpercent, 0);        // Don't print percentage stats
  ret = GPTLsetoption (GPTLabort_on_error, 1); // Abort on any GPTL error

  ret = GPTLinitialize ();                     // Initialize GPTL
  ret = GPTLstart ("total");                   // Time the whole program

  for (i = 0; i < count; ++i)
    sendbuf[i] = iam;

  dest = (iam + 1)%size;
  source = iam - 1;
  if (source < 0)
    source = size - 1;

  printf ("iam %d sending to %d receiving from %d\n", iam, dest, source);
  ret = MPI_Sendrecv (sendbuf, count, MPI_DOUBLE, dest, tag, 
		      recvbuf, count, MPI_DOUBLE, source, tag, 
		      comm, &status);
  ret = MPI_Isend (sendbuf, count, MPI_DOUBLE, dest, tag, 
		   comm, &request);
  ret = MPI_Irecv (recvbuf, count, MPI_DOUBLE, source, tag, 
		   comm, &request);
  ret = MPI_Wait (&request, &status);
  ret = MPI_Barrier (comm);
  ret = MPI_Bcast (sendbuf, count, MPI_DOUBLE, 0, comm);
  ret = MPI_Allreduce (sendbuf, recvbuf, count, MPI_DOUBLE, MPI_SUM, comm);
  ret = MPI_Finalize ();                       // Clean up MPI

  ret = GPTLstop ("total");
  ret = GPTLpr (iam);                          // Print the results
  return 0;
}
